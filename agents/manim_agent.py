import logging
import time
import re
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from agents.base import BaseAgent
from agents.concept_interpreter import ConceptAnalysis, SubConcept
from agents.manim_models import (
    ScenePlan, SceneAction, ManimSceneCode, RenderResult,
    AnimationResult, AnimationConfig, AnimationMetadata
)
from rendering.manim_renderer import ManimRenderer
from config import settings
from joblib import Memory
import ast
import cProfile

genai.configure(api_key=settings.google_api_key)


class ManimAgent(BaseAgent):
    """
    Manim Agent: Transforms structured concept analysis into visual animations
    using scene planning and Manim code generation with <manim> tag extraction.
    """

    def __init__(
        self,
        api_key: str = settings.google_api_key,
        base_url: str = "",
        model: str = settings.reasoning_model,
        output_dir: Path = settings.output_dir,
        config: Optional[AnimationConfig] = None,
        reasoning_tokens: Optional[float] = None,
        reasoning_effort: Optional[str] = None
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, reasoning_tokens=reasoning_tokens, reasoning_effort=reasoning_effort)
        self.gemini_model = genai.GenerativeModel(model)
        self.output_dir = Path(output_dir)
        self.config = config or AnimationConfig()
        self.config.temperature = 0.3  # Reduce temperature for more consistent code generation

        self.renderer = ManimRenderer(
            output_dir=self.output_dir / "scenes",
            quality=self.config.quality,
            background_color=self.config.background_color,
            timeout=self.config.render_timeout,
            max_retries=self.config.max_retries_per_scene
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scene_codes").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "animations").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scenes").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scene_plans").mkdir(parents=True, exist_ok=True)

        # Add caching
        self.memory = Memory(location=self.output_dir / "cache", verbose=0)
        self._generate_scene_codes = self.memory.cache(self._generate_scene_codes)

    def execute(self, concept_analysis: ConceptAnalysis) -> AnimationResult:
        """Execute the Manim animation generation for the given concept analysis"""
        return self.generate_animations(concept_analysis)

    SCENE_PLANNING_PROMPT = """You are a Manim Scene Planning Agent for an educational STEM animation system.

**TASK**: Create detailed scene plans for animating STEM concepts using Manim (Mathematical Animation Engine).

**INPUT CONCEPT ANALYSIS**:
{concept_analysis}

**ANIMATION GUIDELINES**:

1. **Scene Structure**:
   - Create 1–2 scenes per sub-concept (maximum 8 scenes total)
   - Each scene should be 30–45 seconds long
   - Build scenes logically following sub-concept dependencies
   - Start with foundations, progressively add complexity

2. **Visual Design**:
   - Use clear, educational visual style (dark background, bright elements)
   - Include mathematical notation, equations, diagrams
   - Show relationships and transformations visually
   - Use color coding consistently:
     - Blue (#3B82F6) for known/assumed quantities
     - Green (#22C55E) for newly introduced concepts
     - Red (#EF4444) for key/important results or warnings

3. **Consistency & Continuity (VERY IMPORTANT)**:
   - If an illustrative example is used to demonstrate the concept (e.g., a specific array for a sorting algorithm, a fixed probability scenario, a single graph), **use the exact same example and values across all scenes** unless a scene explicitly explores a variant. 
   - Keep element IDs and targets stable across scenes (e.g., "example_array", "disease_prevalence_box") to preserve continuity.
   - Reuse and transform existing elements instead of recreating them where possible.

4. **Animation Types**:
   - write / create: introduce text, equations, axes, or diagrams
   - transform / replace: mathematical transformations, substitutions, rearrangements
   - fade_in / fade_out: introduce or remove elements
   - move / highlight: focus attention
   - grow / shrink: emphasize scale or importance
   - **wait**: insert a timed pause with nothing changing on screen (used for narration beats)

5. **Pacing & Narration Cues**:
   - Animations should be **slow and deliberate**. After each significant action (write, transform, highlight, etc.), insert a **wait** action of 1.5–3.0 seconds for narration.
   - Typical durations (guideline, adjust as needed):
     - write/create (short text/equation): 4–5s
     - transform/replace (equation/diagram): 8-19s
     - move/highlight: 3-5s
     - fade_in/out: 2-5s
     - wait (narration): 2-4s
   - Prefer easing that reads smoothly (ease-in-out). Include `"parameters": {"easing": "ease_in_out"}` when relevant.

6. **Educational Flow**:
   - Start with context/overview
   - Introduce new elements step-by-step
   - Show relationships and connections visually
   - End with key takeaways or summaries, keeping the same example visible to reinforce learning

7. **Element Naming**:
   - Use descriptive, stable targets (e.g., "bayes_equation", "likelihood_label", "frequency_grid") reused across scenes.
   - When transforming, specify `"parameters": {"from": "<old_target>", "to": "<new_target>"}` where helpful.

8. **LaTeX Formatting (IMPORTANT)**:
   - When specifying equations in parameters, use proper LaTeX with DOUBLE braces for subscripts/superscripts
   - ✅ CORRECT: `"equation": "F_{{n}} = F_{{n-1}} + F_{{n-2}}"`
   - ❌ WRONG: `"equation": "F_n = F_{n-1} + F_{n-2}"`
   - Always escape backslashes: `\\frac`, `\\sum`, `\\int`
   - For text in math mode: `\\text{{your text}}`

9. **Vietnamese Language (REQUIRED)**:
   - All descriptions, titles, texts, and parameters must be in Vietnamese. Example: 'Bayes' Theorem: Context & Setup' → 'Định lý Bayes: Ngữ cảnh và Thiết lập'. Use Vietnamese math terms where possible (e.g., 'prior' → 'xác suất tiên nghiệm').

**OUTPUT FORMAT**:
Return ONLY valid JSON matching this exact structure:
{{
    "scene_plans": [
        {{
            "id": "string",
            "title": "string",
            "description": "string",
            "sub_concept_id": "string",
            "actions": [
                {{
                    "action_type": "string",
                    "element_type": "string",
                    "description": "string",
                    "target": "string",
                    "duration": number,
                    "parameters": {{}}
                }}
            ],
            "scene_dependencies": ["string"]
        }}
    ]
}}

**EXAMPLE** for Bayes' Theorem (consistent example across all scenes: medical test with 1% prevalence, 90% sensitivity, 95% specificity):
{{
    "scene_plans": [
        {
            "id": "intro_context",
            "title": "Định lý Bayes: Bối cảnh & Giới thiệu",
            "description": "Giới thiệu ví dụ về xét nghiệm y tế và định nghĩa xác suất tiên nghiệm, độ nhạy, và độ đặc hiệu.",
            "sub_concept_id": "context_prior",
            "actions": [
                {
                    "action_type": "fade_in",
                    "element_type": "text",
                    "description": "Hiển thị tiêu đề 'Định lý Bayes'",
                    "target": "title_text",
                    "duration": 2.0,
                    "parameters": {"text": "Định lý Bayes", "color": "#FFFFFF"}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Tạm dừng lời dẫn sau khi hiển thị tiêu đề",
                    "target": "narration_pause_1",
                    "duration": 2.0,
                    "parameters": {}
                },
                {
                    "action_type": "write",
                    "element_type": "text",
                    "description": "Trình bày ví dụ minh họa nhất quán",
                    "target": "scenario_text",
                    "duration": 5.0,
                    "parameters": {"text": "Ví dụ xét nghiệm y tế: Tỷ lệ mắc bệnh 1%, Độ nhạy 90%, Độ đặc hiệu 95%", "color": "#FFFFFF", "easing": "ease_in_out"}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Tạm dừng để người dẫn giải thích về tỷ lệ mắc bệnh và đặc tính xét nghiệm",
                    "target": "narration_pause_2",
                    "duration": 2.0,
                    "parameters": {}
                },
                {
                    "action_type": "write",
                    "element_type": "math_equation",
                    "description": "Định nghĩa xác suất tiên nghiệm và các đặc tính xét nghiệm bằng mã màu",
                    "target": "definitions",
                    "duration": 6.0,
                    "parameters": {"equation": "P(D)=0.01,\\ \\text{độ nhạy}=0.90,\\ \\text{độ đặc hiệu}=0.95", "color": "#3B82F6", "easing": "ease_in_out"}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Giữ nguyên trước khi chuyển cảnh",
                    "target": "narration_pause_3",
                    "duration": 2.0,
                    "parameters": {}
                },
                {
                    "action_type": "write",
                    "element_type": "diagram",
                    "description": "Vẽ hộp quần thể để làm nền cho ví dụ, hộp này sẽ giữ nguyên trong các cảnh sau",
                    "target": "population_box",
                    "duration": 6.0,
                    "parameters": {"style": "outlined", "color": "#3B82F6", "label": "Quần thể"}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Tạm dừng để củng cố phần giới thiệu",
                    "target": "narration_pause_4",
                    "duration": 3.0,
                    "parameters": {}
                }
            ],
            "scene_dependencies": []
        },
        {
            "id": "equation_intro",
            "title": "Công thức Bayes",
            "description": "Giới thiệu công thức định lý Bayes và liên hệ các thành phần với ví dụ.",
            "sub_concept_id": "bayes_equation",
            "actions": [
                {
                    "action_type": "write",
                    "element_type": "math_equation",
                    "description": "Viết công thức định lý Bayes",
                    "target": "bayes_equation",
                    "duration": 4.0,
                    "parameters": {"equation": "P(D\\mid +)=\\frac{P(+\\mid D)P(D)}{P(+)}", "color": "#22C55E", "easing": "ease_in_out"}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Tạm dừng để người xem đọc công thức",
                    "target": "narration_pause_5",
                    "duration": 2.0,
                    "parameters": {}
                },
                {
                    "action_type": "write",
                    "element_type": "text",
                    "description": "Gán nhãn cho các thành phần: tiên nghiệm, khả năng, bằng chứng, hậu nghiệm",
                    "target": "term_labels",
                    "duration": 5.0,
                    "parameters": {"text": "Tiên nghiệm: P(D) (xanh lam), Khả năng: P(+|D) (xanh lá), Bằng chứng: P(+) (trắng), Hậu nghiệm: P(D|+) (đỏ)", "color": "#FFFFFF"}
                },
                {
                    "action_type": "highlight",
                    "element_type": "math_equation",
                    "description": "Tô màu các phần trong công thức",
                    "target": "bayes_equation",
                    "duration": 3.0,
                    "parameters": {"spans": [{"term": "P(D)", "color": "#3B82F6"}, {"term": "P(+\\mid D)", "color": "#22C55E"}, {"term": "P(+)", "color": "#FFFFFF"}, {"term": "P(D\\mid +)", "color": "#EF4444"}]}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Tạm dừng sau khi gán nhãn",
                    "target": "narration_pause_6",
                    "duration": 2.0,
                    "parameters": {}
                }
            ],
            "scene_dependencies": ["intro_context"]
        },
        {
            "id": "tree_diagram",
            "title": "Cây xác suất",
            "description": "Minh họa các nhánh xác suất tương ứng với các con số trong ví dụ.",
            "sub_concept_id": "likelihood_evidence",
            "actions": [
                {
                    "action_type": "write",
                    "element_type": "diagram",
                    "description": "Vẽ các nhánh D và ¬D từ quần thể",
                    "target": "probability_tree",
                    "duration": 6.0,
                    "parameters": {"branches": [{"label": "D (1%)", "color": "#3B82F6"}, {"label": "¬D (99%)", "color": "#3B82F6"}]}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Tạm dừng để người dẫn giải thích nhánh",
                    "target": "narration_pause_7",
                    "duration": 2.0,
                    "parameters": {}
                },
                {
                    "action_type": "write",
                    "element_type": "diagram",
                    "description": "Thêm các nhánh kết quả xét nghiệm với độ nhạy và độ đặc hiệu",
                    "target": "probability_tree_outcomes",
                    "duration": 6.0,
                    "parameters": {"branches": [{"from": "D", "label": "+ (90%)", "color": "#22C55E"}, {"from": "D", "label": "− (10%)", "color": "#22C55E"}, {"from": "¬D", "label": "+ (5%)", "color": "#22C55E"}, {"from": "¬D", "label": "− (95%)", "color": "#22C55E"}]}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Tạm dừng sau khi thêm kết quả",
                    "target": "narration_pause_8",
                    "duration": 2.0,
                    "parameters": {}
                },
                {
                    "action_type": "highlight",
                    "element_type": "diagram",
                    "description": "Làm nổi bật các nhánh dẫn đến kết quả '+', thể hiện bằng chứng P(+)",
                    "target": "probability_tree_outcomes",
                    "duration": 3.0,
                    "parameters": {"paths": ["D→+", "¬D→+"], "color": "#EF4444"}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Giữ để nhấn mạnh khái niệm 'bằng chứng' P(+)",
                    "target": "narration_pause_9",
                    "duration": 2.0,
                    "parameters": {}
                }
            ],
            "scene_dependencies": ["intro_context", "equation_intro"]
        },
        {
            "id": "frequency_view",
            "title": "Trực quan bằng lưới tần suất",
            "description": "Sử dụng lưới 10.000 điểm để trực quan hóa P(+) và P(D|+) với cùng dữ liệu.",
            "sub_concept_id": "evidence_frequency",
            "actions": [
                {
                    "action_type": "write",
                    "element_type": "diagram",
                    "description": "Tạo lưới 10.000 điểm trong hộp quần thể (duy trì qua các cảnh)",
                    "target": "frequency_grid",
                    "duration": 6.0,
                    "parameters": {"rows": 100, "cols": 100, "color": "#555555", "parent": "population_box"}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Tạm dừng để người dẫn giải thích khung tần suất",
                    "target": "narration_pause_10",
                    "duration": 2.0,
                    "parameters": {}
                },
                {
                    "action_type": "highlight",
                    "element_type": "diagram",
                    "description": "Tô màu 100 điểm mắc bệnh (1%) bằng màu xanh lam",
                    "target": "frequency_grid_D",
                    "duration": 4.0,
                    "parameters": {"count": 100, "color": "#3B82F6"}
                },
                {
                    "action_type": "highlight",
                    "element_type": "diagram",
                    "description": "Trong nhóm D, tô nổi 90 ca dương tính thật bằng màu xanh lá",
                    "target": "frequency_grid_TP",
                    "duration": 4.0,
                    "parameters": {"count": 90, "color": "#22C55E"}
                },
                {
                    "action_type": "highlight",
                    "element_type": "diagram",
                    "description": "Trong nhóm ¬D, tô viền 495 ca dương tính giả (5% của 9.900) bằng màu xanh lá",
                    "target": "frequency_grid_FP",
                    "duration": 5.0,
                    "parameters": {"count": 495, "style": "outline", "color": "#22C55E"}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Tạm dừng để người xem ghi nhớ các con số",
                    "target": "narration_pause_11",
                    "duration": 3.0,
                    "parameters": {}
                }
            ],
            "scene_dependencies": ["intro_context", "equation_intro", "tree_diagram"]
        },
        {
            "id": "posterior_compute",
            "title": "Tính toán P(D|+)",
            "description": "Tính xác suất hậu nghiệm từng bước bằng cùng dữ liệu và công thức Bayes.",
            "sub_concept_id": "posterior_computation",
            "actions": [
                {
                    "action_type": "write",
                    "element_type": "math_equation",
                    "description": "Thay giá trị số vào công thức Bayes",
                    "target": "substitution",
                    "duration": 5.0,
                    "parameters": {"equation": "P(D\\mid +)=\\frac{0.90\\times 0.01}{0.90\\times 0.01 + 0.05\\times 0.99}", "color": "#FFFFFF"}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Tạm dừng để người dẫn giải thích trước khi rút gọn",
                    "target": "narration_pause_12",
                    "duration": 2.0,
                    "parameters": {}
                },
                {
                    "action_type": "transform",
                    "element_type": "math_equation",
                    "description": "Rút gọn tử số và mẫu số",
                    "target": "substitution",
                    "duration": 4.0,
                    "parameters": {"to_equation": "P(D\\mid +)=\\frac{0.009}{0.009+0.0495}", "color": "#FFFFFF", "easing": "ease_in_out"}
                },
                {
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Tạm dừng trước khi ra kết quả cuối cùng",
                    "target": "narration_pause_13",
                    "duration": 2.0,
                    "parameters": {}
                }
            ],
            "scene_dependencies": ["intro_context", "equation_intro", "tree_diagram", "frequency_view"]
        }
    ]
}}
"""

    CODE_GENERATION_PROMPT = """You are a Manim Code Generation Agent. Generate ONLY working, error-free Manim code.

**CRITICAL: Use ONLY valid Manim methods. Invalid methods = instant crash.**

**INPUT SCENE PLAN**:
{scene_plan}

**REQUIRED CODE STRUCTURE**:
```python
from manim import *

class {class_name}(Scene):
    def construct(self):
        self.camera.background_color = "#0f0f0f"
        # Your code here
```

═══════════════════════════════════════════════════════════════════════════════
🔴 CRITICAL METHODS - ONLY USE THESE (Others will CRASH):
═══════════════════════════════════════════════════════════════════════════════

**VALID positioning methods** (ONLY these exist):
  ✅ obj.shift(direction)              # Relative move
  ✅ obj.move_to(point)                # Absolute position [x, y, z]
  ✅ obj.to_edge(edge, buff=0)         # Snap to screen edge (UP, DOWN, LEFT, RIGHT)
  ✅ obj.next_to(other, direction)     # Position relative to another object
  ✅ obj.scale(factor)                 # Scale by factor
  ✅ obj.set_color(color)              # Change color
  ✅ obj.rotate(angle)                 # Rotate by angle
  ✅ group.arrange(direction, buff=0)  # Arrange objects in group

**INVALID methods** (NEVER use - will crash):
  ❌ obj.to_center()                   # WRONG - doesn't exist
  ❌ obj.center()                      # WRONG - doesn't exist  
  ❌ obj.to_origin()                   # WRONG - doesn't exist
  ❌ obj.center_on_screen()            # WRONG - doesn't exist
  ❌ obj.get_center_point()            # WRONG - doesn't exist

**CORRECT alternatives**:
  ✅ obj.move_to([0, 0, 0])            # Center at origin
  ✅ obj.move_to(ORIGIN)               # Center at origin
  ✅ obj.to_edge(UP)                   # Top of screen
  ✅ obj.shift(ORIGIN - obj.get_center())  # Move to origin

═══════════════════════════════════════════════════════════════════════════════
✅ VALID MANIM OBJECTS & METHODS - Complete Reference:
═══════════════════════════════════════════════════════════════════════════════

**Text Objects** (ALL require font="sans-serif"):
  obj = Text("Hello", font="sans-serif", color=WHITE, font_size=36)
  obj = MathTex(r"F = ma", color=BLUE, font_size=44)
  obj = Tex(r"\\frac{{{{a}}}}{{{{b}}}}")

**Shapes** (Basic):
  obj = Circle(radius=1, color=BLUE, fill_opacity=0.5)
  obj = Square(side_length=2, color=GREEN)
  obj = Rectangle(height=2, width=3, color=RED)
  obj = Triangle(color=YELLOW)
  obj = Ellipse(width=3, height=2)
  obj = Arc(radius=1, angle=PI/2)
  obj = Line(start=[0,0,0], end=[1,1,0])
  obj = Arrow(start=[0,0,0], end=[1,0,0], color=RED)
  obj = Polygon([0,0,0], [1,0,0], [1,1,0])

**Styling** (MUST use these exact signatures):
  obj.set_fill(color, opacity)        # opacity is 0-1 (NOT fill_opacity=)
  obj.set_stroke(color, width)        # width in pixels (NOT stroke_width=)
  obj.set_color(color)
  obj.set_opacity(value)              # 0-1
  obj.set_z_index(value)              # Layer depth

**Positioning** (EXACT method names):
  obj.shift(UP * 2)                   # Move by vector
  obj.move_to([0, 2, 0])              # Move to absolute point
  obj.to_edge(UP, buff=0.5)           # Snap to edge
  obj.next_to(other, DOWN, buff=0.3)  # Next to other object
  obj.align_to(other, UP)             # Align with other
  obj.scale(2)                        # Scale by factor
  obj.rotate(PI/4)                    # Rotate by angle
  group.arrange(RIGHT, buff=1)        # Arrange group horizontally

**Grouping**:
  group = VGroup(obj1, obj2, obj3)    # Group objects
  group.add(obj4)                     # Add to group
  group.arrange(DOWN)                 # Arrange vertically

**Animations** (EXACT names):
  Create(obj)                         # Draw shape
  Write(text)                         # Write text
  FadeIn(obj)                         # Fade in
  FadeOut(obj)                        # Fade out
  Transform(obj1, obj2)               # Transform one to another
  ReplacementTransform(obj1, obj2)    # Replace and transform
  Indicate(obj, color=RED)            # Highlight
  Circumscribe(obj)                   # Draw around
  Flash(obj)                          # Flash effect
  obj.animate.shift(UP)               # Animate motion
  obj.animate.scale(2)                # Animate scale

**Getting info** (Methods that return values):
  obj.get_center()                    # [x, y, z]
  obj.get_width()                     # Width in units
  obj.get_height()                    # Height in units
  obj.get_left()                      # Left edge point
  obj.get_right()                     # Right edge point
  obj.get_top()                       # Top edge point
  obj.get_bottom()                    # Bottom edge point

**COMMON CONSTANTS**:
  UP, DOWN, LEFT, RIGHT               # Directions
  ORIGIN = [0, 0, 0]                  # Center
  WHITE, BLACK, BLUE, RED, GREEN, YELLOW, GRAY, etc.  # Colors
  PI, TAU                             # Math constants

═══════════════════════════════════════════════════════════════════════════════
🔴 #1 CRITICAL ERROR - Empty VGroup Positioning:
═══════════════════════════════════════════════════════════════════════════════

❌ WRONG:
    group = VGroup()  # Empty!
    label = Text("Label").next_to(group, UP)  # CRASH - empty VGroup

❌ WRONG:
    group = VGroup().arrange(RIGHT)  # Empty! CRASH on arrange

❌ WRONG:
    text1 = Text("A")
    text2 = Text("B").next_to(text1, DOWN)  # OK so far
    group = VGroup(text1, text2)  # Now they're grouped
    group.arrange(RIGHT)  # CRASH - text2 is already positioned!

✅ CORRECT:
    text1 = Text("A", font="sans-serif")
    text2 = Text("B", font="sans-serif")
    group = VGroup(text1, text2)  # Create group WITH objects
    group.arrange(DOWN, buff=0.5)  # THEN arrange
    label = Text("Group", font="sans-serif").next_to(group, UP)  # THEN position

✅ CORRECT (Absolute positioning - safest):
    obj1 = Circle().move_to([0, 2, 0])
    obj2 = Square().move_to([0, 0, 0])
    obj3 = Rectangle().move_to([0, -2, 0])
    # No relative positioning = no crashes

═══════════════════════════════════════════════════════════════════════════════
✅ COMPLETE WORKING EXAMPLE:
═══════════════════════════════════════════════════════════════════════════════

<manim>
from manim import *

class GravityDemo(Scene):
    def construct(self):
        self.camera.background_color = "#0f0f0f"
        
        # === STEP 1: Create ALL objects ===
        title = Text("Newton's Law", font="sans-serif", color=WHITE, font_size=48)
        
        earth = Circle(radius=0.8, color=BLUE, fill_opacity=0.7)
        moon = Circle(radius=0.3, color=GRAY, fill_opacity=0.7)
        
        eq = MathTex(r"F = G\\frac{{{{m_1 m_2}}}}{{{{r^2}}}}", color=GREEN, font_size=40)
        
        # === STEP 2: Position (all exist - SAFE) ===
        title.to_edge(UP)
        
        earth.move_to([-2, 0, 0])  # Absolute
        moon.move_to([2, 0, 0])    # Absolute
        
        eq.next_to(earth, DOWN, buff=1)  # Relative (earth exists)
        
        # === STEP 3: Animate ===
        self.play(Write(title), run_time=2)
        self.wait(1)
        
        self.play(Create(earth), Create(moon), run_time=2)
        self.wait(1)
        
        arrow = Arrow(earth.get_right(), moon.get_left(), color=RED)
        self.play(Create(arrow), Write(eq), run_time=3)
        self.wait(2)
        
        self.play(
            FadeOut(title),
            FadeOut(earth),
            FadeOut(moon),
            FadeOut(arrow),
            FadeOut(eq),
            run_time=2
        )

</manim>

═══════════════════════════════════════════════════════════════════════════════
📋 MANDATORY CHECKLIST (Verify EVERY item):
═══════════════════════════════════════════════════════════════════════════════

Before generating code:
  ☐ All positioning methods are from the VALID list above
  ☐ No empty VGroups followed by .arrange() or .next_to()
  ☐ All Text objects have font="sans-serif"
  ☐ All MathTex have 4 braces: r"F_{{{{n}}}}"
  ☐ VGroup created WITH objects (not empty)
  ☐ Objects created BEFORE positioning on them
  ☐ No .to_center(), .center(), or similar invalid methods
  ☐ All animations use valid method names
  ☐ rate_func is smooth, linear, rush_into, or rush_from
  ☐ Code wrapped in <manim>...</manim>

═══════════════════════════════════════════════════════════════════════════════
🚫 ABSOLUTE PROHIBITIONS:
═══════════════════════════════════════════════════════════════════════════════

NEVER use these (they CRASH):
  ❌ .to_center()
  ❌ .center()
  ❌ .center_on_screen()
  ❌ .to_origin()
  ❌ .get_center_point()
  ❌ .get_part_by_text()
  ❌ .get_parts_by_text()
  ❌ fill_opacity= (use set_fill instead)
  ❌ stroke_width= (use set_stroke instead)
  ❌ ease_in_out_quad (use smooth)
  ❌ Empty VGroup().arrange()
  ❌ Text without font="sans-serif"
  ❌ Single braces in LaTeX
  ❌ </manim> inside code

═══════════════════════════════════════════════════════════════════════════════

**OUTPUT**: Generate ONLY the Manim code in <manim> tags. No explanations.
Follow the checklist above EXACTLY - every item matters.

**ADDITIONAL RULES FOR ACCURACY**:
- Translate all text to Vietnamese: e.g., "Bayes' Theorem" → "Định lý Bayes", "Pause for narration" → "Tạm dừng để giải thích".
- For LaTeX: ALWAYS use DOUBLE braces for ALL subscripts/superscripts/nested: e.g., F_{{n}} = F_{{n-1}} + F_{{n-2}}. NEVER use single braces.
- For graphs: Always create axes = Axes() first, then graph = axes.plot(lambda x: ..., x_range=[-5, 5, 0.1], color=BLUE). Use 'plot' method.
- Avoid TypeErrors: Never pass unexpected kwargs like 'font' to non-Text objects. ALL Text must have font='sans-serif'.
- Checklist: No invalid methods; Double-check LaTeX; Use Vietnamese strings in Text/MathTex.
"""

    def generate_animations(self, concept_analysis: ConceptAnalysis) -> AnimationResult:
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.time()
        self.logger.info(f"Starting animation generation for concept: {concept_analysis.main_concept}")

        try:
            scene_plans, raw_plans = self._generate_scene_plans(concept_analysis)
            self.logger.info(f"Generated {len(scene_plans)} scene plans")

            plans_filepath = self._save_scene_plans(scene_plans, concept_analysis, raw_plans)
            self.logger.info(f"Scene plans saved to: {plans_filepath}")

            scene_codes = self._generate_scene_codes(scene_plans)
            self.logger.info(f"Generated {len(scene_codes)} scene codes")

            render_results = self._render_scenes(scene_codes)
            self.logger.info(f"Rendered {sum(1 for r in render_results if r.success)}/{len(render_results)} scenes successfully")

            final_animation = self._concatenate_scenes(render_results)
            total_render_time = sum(r.render_time for r in render_results if r.render_time is not None)
            total_duration = sum(r.duration for r in render_results if r.duration is not None)
            generation_time = time.time() - start_time

            result = AnimationResult(
                success=final_animation is not None,
                concept_id=concept_analysis.main_concept.lower().replace(" ", "_"),
                scene_count=len(scene_codes),
                silent_animation_path=str(final_animation) if final_animation else None,
                total_duration=total_duration if total_duration > 0 else None,
                error_message="" if final_animation else "Failed to concatenate scenes",
                scene_plan=scene_plans,
                scene_codes=scene_codes,
                render_results=render_results,
                generation_time=generation_time,
                total_render_time=total_render_time,
                models_used={"reasoning": self.model},
                token_usage=self.get_token_usage()
            )

            self.logger.info(f"Animation generation completed in {generation_time:.2f}s")
            profiler.disable()
            profiler.dump_stats('profile.stats')
            return result

        except Exception as e:
            self.logger.error(f"Animation generation failed: {e}")
            profiler.disable()
            profiler.dump_stats('profile.stats')
            return AnimationResult(
                success=False,
                concept_id=concept_analysis.main_concept.lower().replace(" ", "_"),
                scene_count=0,
                error_message=str(e),
                scene_plan=[],
                scene_codes=[],
                render_results=[],
                generation_time=time.time() - start_time,
                models_used={"reasoning": self.model},
                token_usage=self.get_token_usage()
            )

    def _generate_scene_plans(self, concept_analysis: ConceptAnalysis) -> tuple[List[ScenePlan], Dict[str, Any]]:
        user_message = f"Analyze this STEM concept and create scene plans:\n\n{json.dumps(concept_analysis.model_dump(), indent=2)}"
        try:
            response_json = self._call_llm_structured(
                system_prompt=self.SCENE_PLANNING_PROMPT,
                user_message=user_message,
                temperature=self.config.temperature,
                max_retries=5  # Increased retries
            )
            scene_plans = [ScenePlan(**plan_data) for plan_data in response_json.get("scene_plans", [])]
            return scene_plans, response_json
        except Exception as e:
            self.logger.error(f"Scene planning failed: {e}")
            raise ValueError(f"Failed to generate scene plans: {e}")

    def _save_scene_plans(self, scene_plans: List[ScenePlan], concept_analysis: ConceptAnalysis, response_json: Dict[str, Any]) -> Path:
        safe_name = "".join(c if c.isalnum() else "_" for c in concept_analysis.main_concept.lower())[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_raw_scene_plans_{timestamp}.json"
        filepath = self.output_dir / "scene_plans" / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Raw scene plans output saved to {filepath}")
        return filepath

    @_memory.cache
    def _generate_scene_codes(self, scene_plans: List[ScenePlan]) -> List[ManimSceneCode]:
        scene_codes = []
        self.logger.info(f"Starting parallel code generation for {len(scene_plans)} scenes")

        def generate_single_scene_code(scene_plan: ScenePlan) -> Optional[ManimSceneCode]:
            try:
                self.logger.info(f"Generating code for scene: {scene_plan.title}")
                class_name = self._sanitize_class_name(scene_plan.id)
                
                try:
                    scene_plan_dict = scene_plan.model_dump()
                except AttributeError:
                    scene_plan_dict = scene_plan.dict()
                
                scene_plan_json = json.dumps(scene_plan_dict, indent=2, ensure_ascii=False)
                formatted_prompt = self.CODE_GENERATION_PROMPT.format(
                    scene_plan=scene_plan_json,
                    class_name=class_name,
                    target_duration="25-30"
                )
                response = self._call_llm(
                    system_prompt=formatted_prompt,
                    user_message="Generate the Manim code for the scene plan specified above.",
                    temperature=self.config.temperature,
                    max_retries=5  # Increased
                )
                manim_code, extraction_method = self._extract_manim_code(response)
                if manim_code:
                    self._save_scene_code(scene_plan.id, class_name, manim_code, response)
                    return ManimSceneCode(
                        scene_id=scene_plan.id,
                        scene_name=class_name,
                        manim_code=manim_code,
                        raw_llm_output=response,
                        extraction_method=extraction_method
                    )
                else:
                    self.logger.error(f"Failed to extract Manim code for scene: {scene_plan.id}")
                    return None
            except Exception as e:
                import traceback
                self.logger.error(f"Code generation failed for scene {scene_plan.id}: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return None

        with ThreadPoolExecutor(max_workers=20) as executor:  # Increased workers
            future_to_plan = {executor.submit(generate_single_scene_code, plan): plan for plan in scene_plans}
            for future in as_completed(future_to_plan):
                result = future.result()
                if result:
                    scene_codes.append(result)
        scene_codes.sort(key=lambda x: scene_plans.index(next(p for p in scene_plans if p.id == x.scene_id)))
        self.logger.info(f"Parallel code generation complete: {len(scene_codes)}/{len(scene_plans)} succeeded")
        return scene_codes

    def _extract_manim_code(self, response: str) -> tuple[str, str]:
        manim_pattern = r'<manim>(.*?)</manim>'
        matches = re.findall(manim_pattern, response, re.DOTALL)
        if matches:
            manim_code = self._clean_manim_code(matches[0].strip())
            return manim_code, "tags"
        class_pattern = r'class\s+(\w+)\s*\(\s*Scene\s*\):.*?(?=\n\nclass|\Z)'
        matches = re.findall(class_pattern, response, re.DOTALL)
        if matches:
            class_start = response.find(f"class {matches[0]}(")
            if class_start != -1:
                remaining = response[class_start:]
                next_class = re.search(r'\n\nclass\s+\w+', remaining)
                manim_code = remaining[:next_class.start()] if next_class else remaining
                if "from manim import" not in manim_code:
                    manim_code = "from manim import *\n\n" + manim_code
                manim_code = self._clean_manim_code(manim_code)
                return manim_code.strip(), "parsing"
        if "class" in response and "def construct" in response:
            cleaned = response.strip()
            if not cleaned.startswith("from"):
                cleaned = "from manim import *\n\n" + cleaned
            cleaned = self._clean_manim_code(cleaned)
            return cleaned, "cleanup"
        return "", "failed"

    def _clean_manim_code(self, code: str) -> str:
        code = code.replace('`', '')
        code = re.sub(r'python\n', '', code, flags=re.IGNORECASE)
        code = re.sub(r'\npython', '', code, flags=re.IGNORECASE)
        code = re.sub(r'^```.*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'</manim>', '', code, flags=re.IGNORECASE)
        code = re.sub(r'<manim>', '', code, flags=re.IGNORECASE)
        code = self._fix_latex_in_code(code)
        code = code.replace('¬', r'\neg')
        code = code.replace('∩', r'\cap')
        code = code.replace('∪', r'\cup')
        code = code.replace('∈', r'\in')
        code = code.replace('∀', r'\forall')
        code = code.replace('∃', r'\exists')
        code = re.sub(r'\n{3,}', '\n\n', code)
        code = code.strip()
        code = code.encode('utf-8').decode('utf-8')  # Ensure UTF-8 for Vietnamese
        # Auto-add font to Text if missing
        code = re.sub(r'Text\(([^,]+)\)', r"Text(\1, font='sans-serif')", code)
        # Replace get_graph with plot if present
        code = code.replace('.get_graph(', '.plot(')
        return code

    def _fix_latex_in_code(self, code: str) -> str:
        max_iterations = 20
        for iteration in range(max_iterations):
            original = code
            code = re.sub(r'([_^])\{([^{}]+)\}', r'\1{{\2}}', code)
            code = re.sub(r'\\([a-zA-Z]+)\{([^{}]*?)\}', r'\\\1{{\2}}', code)
            code = re.sub(r'\\text\{([^{}]*?)\}', r'\\text{{\1}}', code)
            code = re.sub(r'\\([a-zA-Z]+)\{([^{}]*)\}', r'\\\1{{\2}}', code)
            if code == original:
                break
        code = code.replace('đ', r'\dj ')  # For Vietnamese accents if needed
        return code

    def _sanitize_class_name(self, scene_id: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', scene_id)
        if sanitized and sanitized[0].isdigit():
            sanitized = "Scene_" + sanitized
        sanitized = sanitized.title().replace('_', '')
        return sanitized if sanitized else "AnimationScene"

    def _save_scene_code(self, scene_id: str, class_name: str, manim_code: str, raw_output: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scene_id}_{class_name}_{timestamp}.py"
        filepath = self.output_dir / "scene_codes" / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Generated Manim code for scene: {scene_id}\n")
            f.write(f"# Class: {class_name}\n")
            f.write(f"# Generated at: {timestamp}\n\n")
            f.write(manim_code)
        raw_filepath = filepath.with_suffix('.raw.txt')
        with open(raw_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Raw LLM output for scene: {scene_id}\n")
            f.write(f"# Class: {class_name}\n")
            f.write(f"# Generated at: {timestamp}\n\n")
            f.write(raw_output)
        return filepath

    def _render_scenes(self, scene_codes: List[ManimSceneCode]) -> List[RenderResult]:
        render_results = []
        with ThreadPoolExecutor(max_workers=20) as executor:  # Parallel rendering
            future_to_code = {}
            for scene_code in scene_codes:
                output_filename = f"{scene_code.scene_id}_{scene_code.scene_name}.mp4"
                future = executor.submit(self._render_single_scene, scene_code, output_filename)
                future_to_code[future] = scene_code
            for future in as_completed(future_to_code):
                result = future.result()
                render_results.append(result)
        return render_results

    def _render_single_scene(self, scene_code: ManimSceneCode, output_filename: str) -> RenderResult:
        self.logger.info(f"Rendering scene: {scene_code.scene_name}")
        try:
            # Syntax check
            ast.parse(scene_code.manim_code)
            for attempt in range(self.config.max_retries_per_scene):
                render_result = self.renderer.render(
                    manim_code=scene_code.manim_code,
                    scene_name=scene_code.scene_name,
                    output_filename=output_filename
                )
                if render_result.success:
                    break
                else:
                    error_msg = render_result.error_message
                    self.logger.warning(f"Render attempt {attempt+1} failed: {error_msg}. Regenerating code...")
                    # Feedback to LLM for fix
                    response = self._call_llm(
                        system_prompt=self.CODE_GENERATION_PROMPT,
                        user_message=f"Fix this error in the code: {error_msg}\nOriginal code:\n{scene_code.manim_code}",
                        temperature=self.config.temperature,
                        max_retries=3
                    )
                    new_code, _ = self._extract_manim_code(response)
                    if new_code:
                        scene_code.manim_code = new_code
                    else:
                        break
            result = RenderResult(
                scene_id=scene_code.scene_id,
                success=render_result.success,
                video_path=render_result.video_path,
                error_message=render_result.error_message,
                duration=render_result.duration,
                resolution=render_result.resolution,
                render_time=render_result.render_time
            )
            if result.success:
                self.logger.info(f"Successfully rendered: {scene_code.scene_name}")
                self.logger.info(f"  Video path: {result.video_path}")
                self.logger.info(f"  Duration: {result.duration}s")
            else:
                self.logger.error(f"Failed to render {scene_code.scene_name}: {result.error_message}")
            return result
        except Exception as e:
            self.logger.error(f"Rendering failed for {scene_code.scene_name}: {e}")
            return RenderResult(
                scene_id=scene_code.scene_id,
                success=False,
                error_message=str(e)
            )

    def _concatenate_scenes(self, render_results: List[RenderResult]) -> Optional[Path]:
        if not render_results:
            self.logger.error("No render results to concatenate")
            return None
        video_paths = []
        for r in render_results:
            if r.success and r.video_path:
                video_path = Path(r.video_path)
                if not video_path.is_absolute():
                    video_path = (Path.cwd() / video_path).resolve()
                if video_path.exists():
                    video_paths.append(video_path)
                else:
                    self.logger.warning(f"Video path does not exist: {video_path}")
        if not video_paths:
            self.logger.error("No successfully rendered scenes with valid video paths to concatenate")
            return None
        self.logger.info(f"Found {len(video_paths)} videos to concatenate")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"animation_{timestamp}.mp4"
            output_path = self.output_dir / "animations" / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                for video_path in video_paths:
                    abs_path = str(video_path.resolve())
                    temp_file.write(f"file '{abs_path}'\n")
                temp_file_path = temp_file.name
            try:
                cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(temp_file_path),
                    "-c", "copy",
                    "-y",
                    str(output_path.resolve())
                ]
                self.logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0 and output_path.exists():
                    self.logger.info(f"Successfully concatenated animation: {output_filename}")
                    return output_path
                else:
                    self.logger.error(f"FFmpeg concatenation failed: {result.stderr}")
                    return None
            finally:
                import os
                os.unlink(temp_file_path)
        except Exception as e:
            self.logger.error(f"Scene concatenation failed: {e}")
            return None

    def get_generation_stats(self) -> Dict[str, Any]:
        return {
            "token_usage": self.get_token_usage(),
            "model_used": self.model,
            "reasoning_tokens": self.reasoning_tokens,
            "config": self.config.model_dump(),
            "renderer_status": "ready" if self.renderer.validate_manim_installation() else "not_ready"
        }