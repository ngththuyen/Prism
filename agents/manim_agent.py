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
        {{
            "id": "intro_context",
            "title": "Bayes' Theorem: Context & Setup",
            "description": "Introduce the medical testing example and define prior, sensitivity, and specificity.",
            "sub_concept_id": "context_prior",
            "actions": [
                {{
                    "action_type": "fade_in",
                    "element_type": "text",
                    "description": "Display title 'Bayes' Theorem'",
                    "target": "title_text",
                    "duration": 2.0,
                    "parameters": {{"text": "Bayes' Theorem", "color": "#FFFFFF"}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Narration pause after title",
                    "target": "narration_pause_1",
                    "duration": 2.0,
                    "parameters": {{}}
                }},
                {{
                    "action_type": "write",
                    "element_type": "text",
                    "description": "Present consistent example scenario",
                    "target": "scenario_text",
                    "duration": 5.0,
                    "parameters": {{"text": "Medical test scenario: Disease prevalence 1%, Sensitivity 90%, Specificity 95%", "color": "#FFFFFF", "easing": "ease_in_out"}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Pause for narrator to explain prevalence and test properties",
                    "target": "narration_pause_2",
                    "duration": 2.0,
                    "parameters": {{}}
                }},
                {{
                    "action_type": "write",
                    "element_type": "math_equation",
                    "description": "Define prior and test properties with color coding",
                    "target": "definitions",
                    "duration": 6.0,
                    "parameters": {{"equation": "P(D)=0.01,\\ \\text{{sensitivity}}=0.90,\\ \\text{{specificity}}=0.95", "color": "#3B82F6", "easing": "ease_in_out"}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Hold before moving on",
                    "target": "narration_pause_3",
                    "duration": 2.0,
                    "parameters": {{}}
                }},
                {{
                    "action_type": "write",
                    "element_type": "diagram",
                    "description": "Draw a population box to anchor the example that persists across scenes",
                    "target": "population_box",
                    "duration": 6.0,
                    "parameters": {{"style": "outlined", "color": "#3B82F6", "label": "Population"}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Pause to reinforce the setup",
                    "target": "narration_pause_4",
                    "duration": 3.0,
                    "parameters": {{}}
                }}
            ],
            "scene_dependencies": []
        }},
        {{
            "id": "equation_intro",
            "title": "Bayes' Formula",
            "description": "Introduce Bayes' theorem and map terms to the example.",
            "sub_concept_id": "bayes_equation",
            "actions": [
                {{
                    "action_type": "write",
                    "element_type": "math_equation",
                    "description": "Write Bayes' theorem",
                    "target": "bayes_equation",
                    "duration": 4.0,
                    "parameters": {{"equation": "P(D\\mid +)=\\frac{{P(+\\mid D)P(D)}}{{P(+)}}", "color": "#22C55E", "easing": "ease_in_out"}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Pause to read equation",
                    "target": "narration_pause_5",
                    "duration": 2.0,
                    "parameters": {{}}
                }},
                {{
                    "action_type": "write",
                    "element_type": "text",
                    "description": "Label terms: prior, likelihood, evidence, posterior",
                    "target": "term_labels",
                    "duration": 5.0,
                    "parameters": {{"text": "prior: P(D) (blue), likelihood: P(+|D) (green), evidence: P(+) (white), posterior: P(D|+) (red)", "color": "#FFFFFF"}}
                }},
                {{
                    "action_type": "highlight",
                    "element_type": "math_equation",
                    "description": "Color-code terms on the formula",
                    "target": "bayes_equation",
                    "duration": 3.0,
                    "parameters": {{"spans": [{{"term": "P(D)", "color": "#3B82F6"}}, {{"term": "P(+\\mid D)", "color": "#22C55E"}}, {{"term": "P(+)", "color": "#FFFFFF"}}, {{"term": "P(D\\mid +)", "color": "#EF4444"}}]}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Narration pause after mapping",
                    "target": "narration_pause_6",
                    "duration": 2.0,
                    "parameters": {{}}
                }}
            ],
            "scene_dependencies": ["intro_context"]
        }},
        {{
            "id": "tree_diagram",
            "title": "Likelihood Paths via Tree",
            "description": "Show a probability tree aligned with the same example numbers.",
            "sub_concept_id": "likelihood_evidence",
            "actions": [
                {{
                    "action_type": "write",
                    "element_type": "diagram",
                    "description": "Draw tree branches for D and ¬D from population",
                    "target": "probability_tree",
                    "duration": 6.0,
                    "parameters": {{"branches": [{{"label": "D (1%)", "color": "#3B82F6"}}, {{"label": "¬D (99%)", "color": "#3B82F6"}}]}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Pause for narrator to explain branches",
                    "target": "narration_pause_7",
                    "duration": 2.0,
                    "parameters": {{}}
                }},
                {{
                    "action_type": "write",
                    "element_type": "diagram",
                    "description": "Add test outcome branches with sensitivity/specificity",
                    "target": "probability_tree_outcomes",
                    "duration": 6.0,
                    "parameters": {{"branches": [{{"from": "D", "label": "+ (90%)", "color": "#22C55E"}}, {{"from": "D", "label": "− (10%)", "color": "#22C55E"}}, {{"from": "¬D", "label": "+ (5%)", "color": "#22C55E"}}, {{"from": "¬D", "label": "− (95%)", "color": "#22C55E"}}]}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Pause after outcomes",
                    "target": "narration_pause_8",
                    "duration": 2.0,
                    "parameters": {{}}
                }},
                {{
                    "action_type": "highlight",
                    "element_type": "diagram",
                    "description": "Highlight the evidence paths that lead to '+'",
                    "target": "probability_tree_outcomes",
                    "duration": 3.0,
                    "parameters": {{"paths": ["D→+", "¬D→+"], "color": "#EF4444"}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Hold to emphasize 'evidence' P(+)",
                    "target": "narration_pause_9",
                    "duration": 2.0,
                    "parameters": {{}}
                }}
            ],
            "scene_dependencies": ["intro_context", "equation_intro"]
        }},
        {{
            "id": "frequency_view",
            "title": "Frequency Grid Intuition",
            "description": "Use a 10,000-dot grid to make P(+) and P(D|+) concrete with the same numbers.",
            "sub_concept_id": "evidence_frequency",
            "actions": [
                {{
                    "action_type": "write",
                    "element_type": "diagram",
                    "description": "Create 10,000-dot grid inside population box (persists across scenes)",
                    "target": "frequency_grid",
                    "duration": 6.0,
                    "parameters": {{"rows": 100, "cols": 100, "color": "#555555", "parent": "population_box"}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Pause for narrator to explain frequency framing",
                    "target": "narration_pause_10",
                    "duration": 2.0,
                    "parameters": {{}}
                }},
                {{
                    "action_type": "highlight",
                    "element_type": "diagram",
                    "description": "Color 100 diseased dots (1%) in blue",
                    "target": "frequency_grid_D",
                    "duration": 4.0,
                    "parameters": {{"count": 100, "color": "#3B82F6"}}
                }},
                {{
                    "action_type": "highlight",
                    "element_type": "diagram",
                    "description": "Among D, highlight 90 true positives in green",
                    "target": "frequency_grid_TP",
                    "duration": 4.0,
                    "parameters": {{"count": 90, "color": "#22C55E"}}
                }},
                {{
                    "action_type": "highlight",
                    "element_type": "diagram",
                    "description": "Among ¬D, highlight 495 false positives (5% of 9,900) in green outline",
                    "target": "frequency_grid_FP",
                    "duration": 5.0,
                    "parameters": {{"count": 495, "style": "outline", "color": "#22C55E"}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Hold to let counts sink in",
                    "target": "narration_pause_11",
                    "duration": 3.0,
                    "parameters": {{}}
                }}
            ],
            "scene_dependencies": ["intro_context", "equation_intro", "tree_diagram"]
        }},
        {{
            "id": "posterior_compute",
            "title": "Compute P(D|+)",
            "description": "Compute the posterior step-by-step using the same counts and Bayes' formula.",
            "sub_concept_id": "posterior_computation",
            "actions": [
                {{
                    "action_type": "write",
                    "element_type": "math_equation",
                    "description": "Substitute numeric values into Bayes' formula",
                    "target": "substitution",
                    "duration": 5.0,
                    "parameters": {{"equation": "P(D\\mid +)=\\frac{{0.90\\times 0.01}}{{0.90\\times 0.01 + 0.05\\times 0.99}}", "color": "#FFFFFF"}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Pause for narrator before simplifying",
                    "target": "narration_pause_12",
                    "duration": 2.0,
                    "parameters": {{}}
                }},
                {{
                    "action_type": "transform",
                    "element_type": "math_equation",
                    "description": "Simplify numerators and denominators",
                    "target": "substitution",
                    "duration": 4.0,
                    "parameters": {{"to_equation": "P(D\\mid +)=\\frac{{0.009}}{{0.009+0.0495}}", "color": "#FFFFFF", "easing": "ease_in_out"}}
                }},
                {{
                    "action_type": "wait",
                    "element_type": "none",
                    "description": "Pause before final result",
                    "target": "narration_pause_13",
                    "duration": 2.0,
                    "parameters": {{}}
                }}
            ],
            "scene_dependencies": ["intro_context", "equation_intro", "tree_diagram", "frequency_view"]
        }}
    ]
}}
"""

    CODE_GENERATION_PROMPT = """You are a Manim Code Generation Agent for an educational STEM animation system.

**TASK**: Generate Manim (Python-based Mathematical Animation Engine) code to visualize a STEM concept based on a provided scene plan.

**INPUT SCENE PLAN**:
{scene_plan}

**MANIM CODE GUIDELINES**:
1. **Code Structure**:
   - Generate a single Manim `Scene` class named `{class_name}` that inherits from `manim.Scene`.
   - Include `from manim import *` at the top.
   - Implement the `construct` method to define all animations for the scene.
   - Ensure the total duration approximates {target_duration} seconds.

2. **Visual Design**:
   - Use a dark background (`#0f0f0f`) and bright elements for visibility.
   - Use consistent color coding:
     - Blue (#3B82F6) for known/assumed quantities.
     - Green (#22C55E) for newly introduced concepts.
     - Red (#EF4444) for key results or warnings.
   - Use `Tex` for mathematical equations, `Text` for regular text, and appropriate Manim objects (e.g., `Dot`, `Line`, `Rectangle`) for diagrams.

3. **Animation Actions**:
   - Map scene plan actions to Manim methods:
     - `write`: Use `Write` for text, equations, or diagrams.
     - `transform`: Use `Transform` or `ReplacementTransform`.
     - `fade_in`: Use `FadeIn`.
     - `fade_out`: Use `FadeOut`.
     - `move`: Use `MoveToTarget` or `ApplyMethod`.
     - `highlight`: Use `Indicate` or `Flash`.
     - `grow`: Use `GrowFromCenter` or `GrowFromPoint`.
     - `shrink`: Use `ShrinkToCenter`.
     - `wait`: Use `self.wait(duration)`.
   - Apply easing via `rate_func` (e.g., `rate_func=smooth` or `rate_func=linear`) when specified in `parameters`. DO NOT use `ease_in_out_quad` - use `smooth` instead.

4. **Consistency**:
   - Reuse element targets (e.g., `bayes_equation`, `population_box`) from the scene plan to maintain continuity.
   - If `parameters` specify `from` and `to` targets for transforms, use `ReplacementTransform(from_obj, to_obj)`.
   - Keep consistent example values (e.g., same array, probabilities) as in the scene plan.

5. **Pacing**:
   - Use durations from the scene plan (e.g., 4–5s for `write`, 2–5s for `fade_in`, 2–4s for `wait`).
   - Insert `self.wait(duration)` after significant actions for narration pauses.
   - Ensure smooth transitions with easing (`ease_in_out` where specified).

6. **Error Handling & Common Mistakes to AVOID**:
   - ❌ DO NOT use `ease_in_out_quad` - use `smooth` instead
   - ❌ DO NOT use external image files (e.g., `ImageMobject("file.png")`) - they don't exist
   - ❌ DO NOT use single-character subscripts in LaTeX like `F_0` - use `F_{{0}}` with double braces
   - ✅ Always wrap subscripts/superscripts in double braces: `F_{{n}}`, `x_{{1}}`, `a^{{2}}`
   - ✅ Use only built-in Manim objects: `Circle`, `Square`, `Dot`, `Line`, `Text`, `MathTex`, `Tex`
   - ✅ For rate_func, use: `smooth`, `linear`, `rush_into`, `rush_from`
   - Ensure code is syntactically correct and runnable in Manim.
   - If an action's `parameters` are unclear, use sensible defaults (e.g., white color, smooth easing).

**OUTPUT FORMAT**:
Return ONLY the Manim code wrapped in <manim> tags:
<manim>
from manim import *

class {class_name}(Scene):
    def construct(self):
        # Your Manim code here
        pass
</manim>

**EXAMPLE** for a scene plan (e.g., Bayes' Theorem introduction):
<manim>
from manim import *

class BayesIntro(Scene):
    def construct(self):
        title = Text("Bayes' Theorem", color=WHITE).to_edge(UP)
        self.play(FadeIn(title), run_time=2)
        self.wait(2)
        scenario = Text("Medical test: Disease prevalence 1%, Sensitivity 90%, Specificity 95%", color=WHITE, font_size=24).shift(UP*2)
        self.play(Write(scenario), run_time=5, rate_func=smooth)
        self.wait(2)
        definitions = MathTex(r"P(D)=0.01,\ \text{sensitivity}=0.90,\ \text{specificity}=0.95", color=BLUE).shift(UP*0.5)
        self.play(Write(definitions), run_time=6, rate_func=smooth)
        self.wait(2)
        population_box = Rectangle(height=3, width=4, color=BLUE, fill_opacity=0).shift(DOWN)
        population_label = Text("Population", color=BLUE, font_size=24).next_to(population_box, UP)
        self.play(Create(population_box), Write(population_label), run_time=6)
        self.wait(3)
</manim>
"""

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
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON parse error on attempt {attempt+1}: {e}")
            except Exception as e:
                self.logger.warning(f"Gemini API error on attempt {attempt+1}: {e}")
        raise ValueError("Failed to get valid JSON response after retries")

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
            self.logger.info(f"Rendered {sum(1 for r in render_results if r.success)}/{len(render_results)} scenes successfully")

            final_animation = self._concatenate_scenes(render_results)
            total_render_time = sum(r.render_time for r in render_results if r.render_time is not None)
            generation_time = time.time() - start_time

            result = AnimationResult(
                success=final_animation is not None,
                concept_id=concept_analysis.main_concept.lower().replace(" ", "_"),
                scene_count=len(scene_codes),
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

    def _generate_scene_plans(self, concept_analysis: ConceptAnalysis) -> tuple[List[ScenePlan], Dict[str, Any]]:
        user_message = f"Analyze this STEM concept and create scene plans:\n\n{json.dumps(concept_analysis.model_dump(), indent=2)}"
        try:
            response_json = self._call_llm_structured(
                system_prompt=self.SCENE_PLANNING_PROMPT,
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
        self.logger.info(f"Raw scene plans output saved to {filepath}")
        return filepath

    def _generate_scene_codes(self, scene_plans: List[ScenePlan]) -> List[ManimSceneCode]:
        scene_codes = []
        self.logger.info(f"Starting parallel code generation for {len(scene_plans)} scenes")

        def generate_single_scene_code(scene_plan: ScenePlan) -> Optional[ManimSceneCode]:
            try:
                self.logger.info(f"Generating code for scene: {scene_plan.title}")
                class_name = self._sanitize_class_name(scene_plan.id)
                scene_plan_json = json.dumps(scene_plan.model_dump(), indent=2)
                formatted_prompt = self.CODE_GENERATION_PROMPT.format(
                    scene_plan=scene_plan_json,
                    class_name=class_name,
                    target_duration="25-30"
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
        raw_filepath = filepath.with_suffix('.raw.txt')
        with open(raw_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Raw LLM output for scene: {scene_id}\n")
            f.write(f"# Class: {class_name}\n")
            f.write(f"# Generated at: {timestamp}\n\n")
            f.write(raw_output)
        return filepath

    def _render_scenes(self, scene_codes: List[ManimSceneCode]) -> List[RenderResult]:
        render_results = []
        for scene_code in scene_codes:
            self.logger.info(f"Rendering scene: {scene_code.scene_name}")
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