import logging
import time
import re
import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
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

genai.configure(api_key=settings.google_api_key)


class ManimCodeValidator:
    """Validate and auto-fix Manim code before rendering"""
    
    # ONLY valid rate functions in Manim (Manim Community v0.19+)
    VALID_RATE_FUNCS = {
        'smooth', 'linear', 'rush_into', 'rush_from', 'there_and_back',
        'ease_in_sine', 'ease_out_sine', 'ease_in_out_sine',
        'ease_in_quad', 'ease_out_quad', 'ease_in_out_quad',
        'ease_in_cubic', 'ease_out_cubic', 'ease_in_out_cubic',
        'ease_in_quart', 'ease_out_quart', 'ease_in_out_quart',
        'ease_in_quint', 'ease_out_quint', 'ease_in_out_quint',
        'ease_in_expo', 'ease_out_expo', 'ease_in_out_expo',
        'ease_in_circ', 'ease_out_circ', 'ease_in_out_circ',
        'ease_in_back', 'ease_out_back', 'ease_in_out_back',
    }
    
    # Invalid methods that LLM commonly uses
    INVALID_METHODS = {
        'to_center': 'move_to(ORIGIN)',
        'center': 'move_to(ORIGIN)',
        'center_on_screen': 'move_to(ORIGIN)',
        'to_origin': 'move_to(ORIGIN)',
        'get_center_point': 'get_center()',
    }
    
    # Invalid rate functions - map to valid ones
    INVALID_RATE_FUNCS = {
        'ease_in': 'smooth',
        'ease_out': 'smooth',
        'ease_in_out': 'smooth',
        'ease_in_out_quad': 'smooth',
        'easeInOut': 'smooth',
        'easeInOutQuad': 'smooth',
    }
    
    @classmethod
    def validate_and_fix(cls, code: str) -> Tuple[str, List[str]]:
        """
        Validate and auto-fix common Manim code errors.
        Returns: (fixed_code, list_of_fixes_applied)
        """
        fixes = []
        
        # Fix 1: Replace invalid methods
        for invalid, valid in cls.INVALID_METHODS.items():
            pattern = rf'\.{invalid}\('
            if re.search(pattern, code):
                code = re.sub(pattern, f'.{valid}(', code)
                fixes.append(f"Replaced .{invalid}() with .{valid}()")
        
        # Fix 2: Replace invalid rate_func values
        for invalid, valid in cls.INVALID_RATE_FUNCS.items():
            pattern = rf'rate_func\s*=\s*{re.escape(invalid)}(?![a-zA-Z_])'
            if re.search(pattern, code):
                code = re.sub(pattern, f'rate_func={valid}', code)
                fixes.append(f"Replaced rate_func={invalid} with rate_func={valid}")
        
        # Fix 3: Remove undefined easing function references
        undefined_easing = re.findall(r'rate_func\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)', code)
        for func_name in set(undefined_easing):
            if func_name not in cls.VALID_RATE_FUNCS:
                code = re.sub(rf'rate_func\s*=\s*{func_name}(?![a-zA-Z_])', 'rate_func=smooth', code)
                fixes.append(f"Replaced undefined rate_func={func_name} with rate_func=smooth")
        
        # Fix 4: Ensure all Text has font="sans-serif"
        # Match Text(...) without font= parameter
        text_pattern = r'Text\s*\(\s*(["\'][^"\']*["\']\s*(?:,\s*(?!font=)[a-zA-Z_]+\s*=)?[^)]*)\s*\)'
        def add_font(match):
            content = match.group(1)
            if 'font=' not in content:
                # Insert font after the text string
                return f'Text({content}, font="sans-serif")'
            return match.group(0)
        
        original_code = code
        # Simpler approach: find Text calls without font
        while 'Text("' in code or "Text('" in code:
            # Find first Text( without font parameter nearby
            text_start = code.find('Text(')
            if text_start == -1:
                break
            
            paren_count = 0
            i = text_start + 5
            text_end = -1
            while i < len(code):
                if code[i] == '(':
                    paren_count += 1
                elif code[i] == ')':
                    if paren_count == 0:
                        text_end = i
                        break
                    paren_count -= 1
                i += 1
            
            if text_end == -1:
                break
            
            text_section = code[text_start:text_end+1]
            if 'font=' not in text_section and text_section.startswith('Text('):
                # Add font parameter
                insert_pos = text_end
                code = code[:insert_pos] + ', font="sans-serif"' + code[insert_pos:]
                fixes.append("Added font=\"sans-serif\" to Text object")
            else:
                break
        
        # Fix 5: Fix VGroup().arrange() pattern (empty VGroup)
        # This is harder to auto-fix - just warn
        if re.search(r'VGroup\(\s*\)\.(?:arrange|next_to)', code):
            fixes.append("WARNING: Empty VGroup found with positioning - may crash")
        
        return code, fixes
    
    @classmethod
    def validate_structure(cls, code: str) -> Tuple[bool, List[str]]:
        """Check code structure for critical issues."""
        errors = []
        
        if 'def construct(self)' not in code:
            errors.append("Missing 'def construct(self):' method")
        
        if 'class' not in code or 'Scene' not in code:
            errors.append("Missing Scene class definition")
        
        return len(errors) == 0, errors


class ManimAgent(BaseAgent):
    """
    Manim Agent: Transforms structured concept analysis into visual animations
    using scene planning and Manim code generation with validation.
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
        super().__init__(api_key=api_key, base_url=base_url, model=model, 
                        reasoning_tokens=reasoning_tokens, reasoning_effort=reasoning_effort)
        self.gemini_model = genai.GenerativeModel(model)
        self.output_dir = Path(output_dir)
        self.config = config or AnimationConfig()

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

    def execute(self, concept_analysis: ConceptAnalysis) -> AnimationResult:
        """Execute the Manim animation generation for the given concept analysis"""
        return self.generate_animations(concept_analysis)

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
CRITICAL METHODS - ONLY USE THESE (Others will CRASH)
═══════════════════════════════════════════════════════════════════════════════

VALID positioning methods (ONLY these exist):
  ✅ obj.shift(direction)              # Relative move
  ✅ obj.move_to(point)                # Absolute position [x, y, z]
  ✅ obj.to_edge(edge, buff=0)         # Snap to screen edge (UP, DOWN, LEFT, RIGHT)
  ✅ obj.next_to(other, direction)     # Position relative to another object
  ✅ obj.scale(factor)                 # Scale by factor
  ✅ obj.set_color(color)              # Change color
  ✅ obj.rotate(angle)                 # Rotate by angle
  ✅ group.arrange(direction, buff=0)  # Arrange objects in group

INVALID methods (NEVER use - will crash):
  ❌ obj.to_center()                   # WRONG - doesn't exist
  ❌ obj.center()                      # WRONG - doesn't exist  
  ❌ obj.to_origin()                   # WRONG - doesn't exist
  ❌ obj.center_on_screen()            # WRONG - doesn't exist
  ❌ obj.get_center_point()            # WRONG - doesn't exist

═══════════════════════════════════════════════════════════════════════════════
VALID TEXT & SHAPES
═══════════════════════════════════════════════════════════════════════════════

Text Objects (ALL require font="sans-serif"):
  obj = Text("Hello", font="sans-serif", color=WHITE, font_size=36)
  obj = MathTex(r"F = ma", color=BLUE, font_size=44)
  obj = Tex(r"\\\\frac{{{{a}}}}{{{{b}}}}")

Shapes (Basic):
  obj = Circle(radius=1, color=BLUE, fill_opacity=0.5)
  obj = Square(side_length=2, color=GREEN)
  obj = Rectangle(height=2, width=3, color=RED)
  obj = Triangle(color=YELLOW)
  obj = Line(start=[0,0,0], end=[1,1,0])
  obj = Arrow(start=[0,0,0], end=[1,0,0], color=RED)

Styling (EXACT signatures):
  obj.set_fill(color, opacity)        # opacity is 0-1
  obj.set_stroke(color, width)        # width in pixels
  obj.set_color(color)
  obj.set_opacity(value)              # 0-1

═══════════════════════════════════════════════════════════════════════════════
CRITICAL ERROR PATTERN - #1 cause of crashes
═══════════════════════════════════════════════════════════════════════════════

❌ WRONG - Empty VGroup:
    group = VGroup()
    label = Text("Label").next_to(group, UP)  # CRASH

❌ WRONG - Arrange before adding:
    group = VGroup().arrange(RIGHT)  # Empty! CRASH

✅ CORRECT - Create with objects:
    obj1 = Circle()
    obj2 = Square()
    group = VGroup(obj1, obj2)  # WITH objects
    group.arrange(DOWN, buff=0.5)
    label = Text("Label").next_to(group, UP)

✅ CORRECT - Absolute positioning (safest):
    obj1 = Circle().move_to([0, 2, 0])
    obj2 = Square().move_to([0, 0, 0])
    obj3 = Rectangle().move_to([0, -2, 0])

═══════════════════════════════════════════════════════════════════════════════
VALID RATE FUNCTIONS ONLY
═══════════════════════════════════════════════════════════════════════════════

VALID:
  ✅ rate_func=smooth       # Ease in/out (BEST for education)
  ✅ rate_func=linear       # Constant speed
  ✅ rate_func=rush_into    # Fast start, slow end
  ✅ rate_func=rush_from    # Slow start, fast end

INVALID (DON'T USE - will crash):
  ❌ rate_func=ease_in
  ❌ rate_func=ease_out
  ❌ rate_func=ease_in_out
  ❌ rate_func=ease_in_out_quad
  ❌ rate_func=easeInOut

═══════════════════════════════════════════════════════════════════════════════
COMPLETE WORKING EXAMPLE
═══════════════════════════════════════════════════════════════════════════════

<manim>
from manim import *

class GravityDemo(Scene):
    def construct(self):
        self.camera.background_color = "#0f0f0f"
        
        # Step 1: Create ALL objects first
        title = Text("Newton's Law", font="sans-serif", color=WHITE, font_size=48)
        earth = Circle(radius=0.8, color=BLUE, fill_opacity=0.7)
        moon = Circle(radius=0.3, color=GRAY, fill_opacity=0.7)
        eq = MathTex(r"F = G\\frac{{{{m_1 m_2}}}}{{{{r^2}}}}", color=GREEN, font_size=40)
        
        # Step 2: Position (all exist - SAFE)
        title.to_edge(UP)
        earth.move_to([-2, 0, 0])
        moon.move_to([2, 0, 0])
        eq.next_to(earth, DOWN, buff=1)
        
        # Step 3: Animate
        self.play(Write(title), run_time=2)
        self.wait(1)
        self.play(Create(earth), Create(moon), run_time=2)
        self.wait(1)
        arrow = Arrow(earth.get_right(), moon.get_left(), color=RED)
        self.play(Create(arrow), Write(eq), run_time=3, rate_func=smooth)
        self.wait(2)
        self.play(
            FadeOut(title), FadeOut(earth), FadeOut(moon),
            FadeOut(arrow), FadeOut(eq), run_time=2
        )
</manim>

═══════════════════════════════════════════════════════════════════════════════

**OUTPUT**: Generate ONLY the Manim code in <manim> tags. No explanations."""

    def _call_llm(self, system_prompt: str, user_message: str, temperature: float = 0.5, max_retries: int = 3) -> str:
        prompt = f"{system_prompt}\n\nUser: {user_message}"
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature}
                )
                return response.text.strip()
            except Exception as e:
                self.logger.warning(f"Gemini API error on attempt {attempt+1}: {e}")
        raise ValueError("Failed to get valid response after retries")

    def _call_llm_structured(self, system_prompt: str, user_message: str, temperature: float = 0.5, max_retries: int = 3) -> Dict:
        prompt = f"{system_prompt}\n\nUser: {user_message}"
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature}
                )
                response_text = response.text.strip()
                
                if response_text.startswith('```json'):
                    response_text = response_text[7:].strip()
                if response_text.endswith('```'):
                    response_text = response_text[:-3].strip()
                
                response_text = self._fix_latex_escapes_in_json(response_text)
                return json.loads(response_text, strict=False)
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON parse error on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed JSON (first 500 chars): {response_text[:500]}")
            except Exception as e:
                self.logger.warning(f"Gemini API error on attempt {attempt+1}: {e}")
        raise ValueError("Failed to get valid JSON response after retries")
    
    def _fix_latex_escapes_in_json(self, text: str) -> str:
        """Fix LaTeX escape sequences in JSON strings"""
        protected = {}
        counter = 0
        
        for escape in [r'\"', r'\\', r'\/', r'\b', r'\f', r'\n', r'\r', r'\t']:
            placeholder = f"__JSON_ESCAPE_{counter}__"
            protected[placeholder] = escape
            text = text.replace(escape, placeholder)
            counter += 1
        
        text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: f"__JSON_UNICODE_{m.group(1)}__", text)
        text = text.replace('\\', '\\\\')
        
        for placeholder, original in protected.items():
            text = text.replace(placeholder, original)
        
        text = re.sub(r'__JSON_UNICODE_([0-9a-fA-F]{4})__', r'\\u\1', text)
        return text

    def validate_and_fix_code(self, manim_code: str) -> Tuple[str, List[str], bool]:
        """
        Validate and auto-fix Manim code before rendering.
        Returns: (fixed_code, list_of_fixes, is_valid)
        """
        fixed_code, fixes = ManimCodeValidator.validate_and_fix(manim_code)
        is_valid, errors = ManimCodeValidator.validate_structure(fixed_code)
        
        if fixes:
            self.logger.info(f"Applied {len(fixes)} code fixes:")
            for fix in fixes:
                self.logger.info(f"  • {fix}")
        
        if errors:
            self.logger.warning(f"Code validation warnings ({len(errors)}):")
            for error in errors:
                self.logger.warning(f"  • {error}")
        
        return fixed_code, fixes, is_valid

    def generate_animations(self, concept_analysis: ConceptAnalysis) -> AnimationResult:
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
            success_count = sum(1 for r in render_results if r.success)
            self.logger.info(f"Rendered {success_count}/{len(render_results)} scenes successfully")

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
            return result

        except Exception as e:
            self.logger.error(f"Animation generation failed: {e}")
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

    def _generate_scene_plans(self, concept_analysis: ConceptAnalysis) -> Tuple[List[ScenePlan], Dict[str, Any]]:
        user_message = f"Analyze this STEM concept and create scene plans:\n\n{json.dumps(concept_analysis.model_dump(), indent=2)}"
        try:
            response_json = self._call_llm_structured(
                system_prompt="Generate detailed scene plans for STEM animation.",
                user_message=user_message,
                temperature=self.config.temperature,
                max_retries=3
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
        return filepath

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
                    target_duration="20-25"
                )
                response = self._call_llm(
                    system_prompt=formatted_prompt,
                    user_message="Generate the Manim code for the scene plan specified above.",
                    temperature=self.config.temperature,
                    max_retries=3
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
                self.logger.error(f"Code generation failed for scene {scene_plan.id}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=min(len(scene_plans), 10)) as executor:
            future_to_plan = {executor.submit(generate_single_scene_code, plan): plan for plan in scene_plans}
            for future in as_completed(future_to_plan):
                result = future.result()
                if result:
                    scene_codes.append(result)
        
        scene_codes.sort(key=lambda x: scene_plans.index(next(p for p in scene_plans if p.id == x.scene_id)))
        self.logger.info(f"Code generation complete: {len(scene_codes)}/{len(scene_plans)} succeeded")
        return scene_codes

    def _extract_manim_code(self, response: str) -> Tuple[str, str]:
        manim_pattern = r'<manim>(.*?)</manim>'
        matches = re.findall(manim_pattern, response, re.DOTALL)
        if matches:
            manim_code = self._clean_manim_code(matches[0].strip())
            return manim_code, "tags"
        
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
        code = re.sub(r'\n{3,}', '\n\n', code)
        code = code.strip()
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
        return filepath

    def _render_scenes(self, scene_codes: List[ManimSceneCode]) -> List[RenderResult]:
        render_results = []
        for scene_code in scene_codes:
            self.logger.info(f"Rendering scene: {scene_code.scene_name}")
            
            # NEW: Validate and fix code before rendering
            fixed_code, fixes, is_valid = self.validate_and_fix_code(scene_code.manim_code)
            if fixes:
                self.logger.info(f"Applied {len(fixes)} auto-fixes to scene code")
                scene_code.manim_code = fixed_code
            
            if not is_valid:
                self.logger.warning(f"Scene has structure issues, attempting render anyway")
            
            output_filename = f"{scene_code.scene_id}_{scene_code.scene_name}.mp4"
            try:
                render_result = self.renderer.render(
                    manim_code=scene_code.manim_code,
                    scene_name=scene_code.scene_name,
                    output_filename=output_filename
                )
                result = RenderResult(
                    scene_id=scene_code.scene_id,
                    success=render_result.success,
                    video_path=render_result.video_path,
                    error_message=render_result.error_message,
                    duration=render_result.duration,
                    resolution=render_result.resolution,
                    render_time=render_result.render_time
                )
                render_results.append(result)
                if result.success:
                    self.logger.info(f"Successfully rendered: {scene_code.scene_name}")
                    self.logger.info(f"  Video path: {result.video_path}")
                    self.logger.info(f"  Duration: {result.duration}s")
                else:
                    self.logger.error(f"Failed to render {scene_code.scene_name}: {result.error_message}")
            except Exception as e:
                self.logger.error(f"Rendering failed for {scene_code.scene_name}: {e}")
                render_results.append(RenderResult(
                    scene_id=scene_code.scene_id,
                    success=False,
                    error_message=str(e)
                ))
        return render_results

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