"""
manim_agent.py

Refactored ManimAgent that tightens LLM output to valid Manim code
by validating against a whitelist of allowed objects/methods and applying
safe auto-fixes for common problems (LaTeX braces, forbidden methods,
empty VGroup usage, Text font missing, missing imports, <manim> tags, ...).

Usage: replace your existing manim_agent.py with this file (or import
ManimAgent from it). The agent still calls a generative model but will
reject or auto-fix outputs that violate the whitelist before saving or
rendering.

This file was produced as an improved replacement; the original file you
uploaded was used as reference for integration points and design.
"""

import re
import json
import logging
import time
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# NOTE: keep external dependencies minimal here; adapt imports to your project
# The following imports should exist in your project; if not, swap with stubs
try:
    import google.generativeai as genai
except Exception:
    genai = None  # allow tests without API

from agents.base import BaseAgent
from agents.concept_interpreter import ConceptAnalysis
from agents.manim_models import (
    ScenePlan, ManimSceneCode, AnimationResult, RenderResult, AnimationConfig
)
from rendering.manim_renderer import ManimRenderer
from config import settings

# Configure API if available
if genai is not None:
    genai.configure(api_key=settings.google_api_key)

logger = logging.getLogger(__name__)


# ----------------------------- Validator / Fixer -----------------------------

ALLOWED_IDENTIFIERS = {
    # Objects
    "Text", "MathTex", "Tex", "Circle", "Square", "Rectangle", "Triangle",
    "Ellipse", "Arc", "Line", "Arrow", "Polygon", "VGroup",
    # Animations
    "Create", "Write", "FadeIn", "FadeOut", "Transform", "ReplacementTransform",
    "Indicate", "Circumscribe", "Flash",
    # Methods (positioning / styling / info)
    "shift", "move_to", "to_edge", "next_to", "scale", "set_color",
    "rotate", "arrange", "set_fill", "set_stroke", "set_opacity", "set_z_index",
    "animate", "get_center", "get_width", "get_height", "get_left", "get_right",
    "get_top", "get_bottom", "align_to",
    # Constants
    "UP", "DOWN", "LEFT", "RIGHT", "ORIGIN", "PI", "TAU", "WHITE", "BLACK",
}

FORBIDDEN_PATTERNS = [
    r"\.to_center\s*\(|\.center\s*\(|\.center_on_screen\s*\(|\.to_origin\s*\(|\.get_center_point\s*\(|",
    r"fill_opacity\s*=", r"stroke_width\s*="
]

MANDATORY_CHECKS = [
    "Text_font",
    "MathTex_braces",
    "No_empty_vgroup_arrange",
    "Valid_identifiers_only",
    "Has_import_from_manim"
]


def find_manim_tag_block(text: str) -> Optional[str]:
    """Extract content inside <manim>...</manim> if present, else None."""
    m = re.search(r"<manim>(.*?)</manim>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def ensure_manim_tags(code: str) -> str:
    """Wrap code in <manim> tags if not present. (Used only for storage/LLM contract.)"""
    if find_manim_tag_block(code):
        return code
    return f"<manim>\n{code}\n</manim>"


def extract_python_from_response(text: str) -> str:
    """Try to extract python class definition or code fences from LLM response."""
    # Common: code inside ```python ... ``` or ``` ... ``` or raw <manim> tags
    if find_manim_tag_block(text):
        return find_manim_tag_block(text)

    # Extract triple-backtick blocks
    m = re.search(r"```(?:python)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: find 'class Xxx(Scene):' through the end or before another class
    m = re.search(r"(class\s+\w+\s*\(\s*Scene\s*\)\s*:\s*.*)$", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # As last resort, return full text (will be validated)
    return text.strip()


def simple_tokenize_identifiers(code: str) -> List[str]:
    """Return candidate identifiers used in code (callables and names). Not perfect but good enough."""
    # This finds capitalized names (classes/objects) and function names
    tokens = set()
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", code):
        tokens.add(m.group(1))
    return sorted(tokens)


def contains_forbidden_patterns(code: str) -> List[str]:
    matches = []
    for patt in FORBIDDEN_PATTERNS:
        if re.search(patt, code):
            matches.append(patt)
    return matches


def ensure_text_font(code: str) -> Tuple[str, bool]:
    """Ensure Text(...) always includes font="sans-serif". Return (new_code, changed)."""
    changed = False

    def repl(m):
        nonlocal changed
        inner = m.group(0)
        # if font= already present, keep
        if re.search(r"\bfont\s*=", inner):
            return inner
        changed = True
        # inject font="sans-serif" before the closing ) or before other kwargs
        return inner[:-1] + ", font=\"sans-serif\")"

    code_new = re.sub(r"Text\([^\)]*\)", repl, code)
    return code_new, changed


def ensure_mathtex_braces(code: str) -> Tuple[str, bool]:
    """Convert single-braced LaTeX to doubled braces inside raw strings for MathTex/Tex.
    Only operates inside r"..." or r'...'."""
    changed = False

    def fix_inner(s: str) -> str:
        # Replace { ... } by {{ ... }} but avoid already doubled
        nonlocal changed
        original = s
        # skip already doubled braces
        s = re.sub(r"(?<!\{)\{([^\{\}]+)\}(?!\})", r"{{\1}}", s)
        if s != original:
            changed = True
        return s

    # handle MathTex(r"...") and Tex(r"...")
    def repl(m):
        prefix = m.group(1)
        quote = m.group(2)
        inner = m.group(3)
        fixed = fix_inner(inner)
        return f"{prefix}{quote}{fixed}{quote}"

    patt = re.compile(r"(MathTex\s*\(|Tex\s*\()([rR]?[\"\'])(.*?)([\"\'])\)", re.DOTALL)
    # simpler pass: find MathTex\(r"..."\)
    code_new = patt.sub(lambda mm: mm.group(1) + mm.group(2) + fix_inner(mm.group(3)) + mm.group(4) + ")", code)

    # If nothing matched, return unchanged
    return code_new, changed


def detect_empty_vgroup_arrange(code: str) -> bool:
    """Detect patterns like VGroup().arrange or VGroup().next_to
    or VGroup() followed by .arrange on a separate line."""
    # direct: VGroup().arrange(
    if re.search(r"VGroup\s*\(\s*\)\s*\.\s*arrange\s*\(", code):
        return True
    # separate creation then immediate arrange after nothing added: group = VGroup()\ngroup.arrange
    m = re.search(r"(\w+)\s*=\s*VGroup\s*\(\s*\)\s*\n\s*\1\s*\.\s*arrange\s*\(", code)
    return bool(m)


def find_missing_import_from_manim(code: str) -> bool:
    return not bool(re.search(r"from\s+manim\s+import\s+\*", code))


def validate_code_against_whitelist(code: str) -> Tuple[bool, List[str]]:
    """Return (is_valid, problems_list)."""
    problems = []

    # 1) Forbidden patterns
    forbidden = contains_forbidden_patterns(code)
    if forbidden:
        problems.append("forbidden_patterns: " + ";".join(forbidden))

    # 2) Empty VGroup arrange
    if detect_empty_vgroup_arrange(code):
        problems.append("empty_vgroup_arrange")

    # 3) Missing import
    if find_missing_import_from_manim(code):
        problems.append("missing_from_manim_import")

    # 4) Text font
    if re.search(r"Text\([^\)]*\)", code) and not re.search(r"Text\([^\)]*font\s*=", code):
        problems.append("text_missing_font")

    # 5) MathTex braces - check for MathTex or Tex with single braces
    # We do a simple heuristic: look for MathTex(r"...{...}...") where braces not doubled
    if re.search(r"MathTex\s*\(r?[\"\'].*\{[^\{\}]*\}.*[\"\']\)", code):
        # if we find already double braces, it's OK; else mark
        if not re.search(r"MathTex\s*\(r?[\"\'].*\{\{[^\}]+\}\}.*[\"\']\)", code):
            problems.append("mathtex_single_braces")

    # 6) Identifier whitelist - flag any capitalized identifier not in allowed set
    tokens = simple_tokenize_identifiers(code)
    unknowns = [t for t in tokens if t[0].isupper() and t not in ALLOWED_IDENTIFIERS]
    # reduce false positives by allowing class names starting with Scene or custom Scene classes
    unknowns = [u for u in unknowns if not re.match(r"Scene|Scene\w+", u)]
    if unknowns:
        problems.append("unknown_identifiers:" + ",".join(unknowns[:10]))

    is_valid = len(problems) == 0
    return is_valid, problems


def attempt_auto_fix(code: str) -> Tuple[str, List[str], bool]:
    """Try several safe auto-fixes. Return (new_code, applied_fixes_list, changed_flag)."""
    fixes = []
    changed = False
    new_code = code

    # Fix 1: Ensure from manim import * exists (safe to add at top)
    if find_missing_import_from_manim(new_code):
        new_code = "from manim import *\n\n" + new_code
        fixes.append("added_from_manim_import")
        changed = True

    # Fix 2: Ensure Text has font
    new_code, c = ensure_text_font(new_code)
    if c:
        fixes.append("text_font_injected")
        changed = True

    # Fix 3: Attempt to fix MathTex/Tex braces
    new_code, c2 = ensure_mathtex_braces(new_code)
    if c2:
        fixes.append("mathtex_braces_doubled")
        changed = True

    # Fix 4: Remove obvious forbidden method calls by replacing with safer alternatives
    for patt, repl in [
        (r"\.to_center\s*\(", ".move_to(ORIGIN)"),
        (r"\.center_on_screen\s*\(", ".move_to(ORIGIN)"),
        (r"fill_opacity\s*=", "set_fill("),
        (r"stroke_width\s*=", "set_stroke(")
    ]:
        if re.search(patt, new_code):
            new_code = re.sub(patt, repl, new_code)
            fixes.append(f"replaced_{patt}")
            changed = True

    # Fix 5: Prevent empty VGroup().arrange() by rewriting simple pattern to create safe placeholders
    # Replace `group = VGroup()` followed soon by `group.arrange(` with a safe comment warning
    new_code = re.sub(r"(\w+)\s*=\s*VGroup\s*\(\s*\)\s*\n\s*(\1\s*\.\s*arrange\s*\()",
                      r"\1 = VGroup()  # NOTE: verify objects are added before calling arrange\n\2",
                      new_code)

    return new_code, fixes, changed


# ----------------------------- ManimAgent class -----------------------------

class ManimAgent(BaseAgent):
    """ManimAgent that generates scene plans, calls an LLM to produce Manim code,
    and — critically — validates and auto-fixes the generated code before saving
    or rendering. If validation still fails after auto-fix, the agent will refuse
    to use the code and return a helpful error structure.
    """

    def __init__(
        self,
        api_key: str = settings.google_api_key,
        model: str = settings.reasoning_model,
        output_dir: Path = settings.output_dir,
        config: Optional[AnimationConfig] = None,
        reasoning_tokens: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ):
        super().__init__(api_key=api_key, base_url="", model=model, reasoning_tokens=reasoning_tokens, reasoning_effort=reasoning_effort)
        self.gemini_model = genai.GenerativeModel(model) if genai is not None else None
        self.output_dir = Path(output_dir)
        self.config = config or AnimationConfig()

        self.renderer = ManimRenderer(
            output_dir=self.output_dir / "scenes",
            quality=self.config.quality,
            background_color=self.config.background_color,
            timeout=self.config.render_timeout,
            max_retries=self.config.max_retries_per_scene
        )

        # ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "scene_codes").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "animations").mkdir(parents=True, exist_ok=True)

    # ----------- LLM wrappers (kept small; adapt to your project's LLM client) -----------
    def _call_llm(self, system_prompt: str, user_message: str, temperature: float = 0.0, max_retries: int = 2) -> str:
        prompt = f"{system_prompt}\n\nUser: {user_message}"
        if self.gemini_model is None:
            raise RuntimeError("LLM client not configured")
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(prompt, generation_config={"temperature": temperature})
                return response.text.strip()
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt+1} failed: {e}")
                time.sleep(0.5 + attempt)
        raise RuntimeError("LLM failed after retries")

    # ---------------- Code extraction + validation ----------------
    def _extract_manim_code(self, llm_response: str) -> Tuple[str, Dict[str, Any]]:
        """Extract python code snippet and run validation + auto-fix. Returns (final_code, meta).
        meta contains keys: raw, extracted, fixes, valid, problems.
        """
        meta: Dict[str, Any] = {"raw": llm_response}
        extracted = extract_python_from_response(llm_response)
        meta["extracted"] = extracted[:5000]

        code = extracted
        # Clean out stray HTML tags
        code = re.sub(r"</?manim>", "", code, flags=re.IGNORECASE)
        code = code.strip()

        # Run validation
        valid, problems = validate_code_against_whitelist(code)
        meta["valid_before"] = valid
        meta["problems_before"] = problems

        if valid:
            meta["final_code"] = ensure_manim_tags(code)
            meta["fixes"] = []
            return meta["final_code"], meta

        # Attempt auto-fix
        fixed_code, fixes, changed = attempt_auto_fix(code)
        meta["fixes"] = fixes
        meta["changed_by_fixer"] = changed

        # Re-validate
        valid_after, problems_after = validate_code_against_whitelist(fixed_code)
        meta["valid_after"] = valid_after
        meta["problems_after"] = problems_after

        if valid_after:
            meta["final_code"] = ensure_manim_tags(fixed_code)
            return meta["final_code"], meta

        # If still invalid, produce a sanitized wrapper that contains a safe placeholder
        # so system does not try to render unsafe code. We still save the LLM output and meta.
        placeholder = (
            "<manim>\n# CODE REJECTED BY VALIDATOR\nfrom manim import *\n\n" +
            "class ValidationFailedScene(Scene):\n" +
            "    def construct(self):\n" +
            "        warning = Text(\"Generated code failed validator - see logs\", font=\"sans-serif\")\n" +
            "        self.play(Write(warning))\n" +
            "\n</manim>"
        )
        meta["final_code"] = placeholder
        return placeholder, meta

    # ----------------------- Helpers to save & render -----------------------
    def _save_scene_code(self, scene_id: str, class_name: str, manim_code: str, meta: Dict[str, Any]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scene_id}_{class_name}_{timestamp}.py"
        filepath = self.output_dir / "scene_codes" / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Generated Manim code for scene: {scene_id}\n")
            f.write(f"# Class: {class_name}\n")
            f.write(f"# Generated at: {timestamp}\n")
            f.write(f"# Validator meta: {json.dumps({k: v for k, v in meta.items() if k!='raw'}, ensure_ascii=False)[:1000]}\n\n")
            f.write(manim_code)
        raw_filepath = filepath.with_suffix('.raw.txt')
        with open(raw_filepath, 'w', encoding='utf-8') as rf:
            rf.write(meta.get('raw', ''))
        return filepath

    # ----------------------- High-level pipeline funcs -----------------------
    def generate_scene_code_for_plan(self, scene_plan: ScenePlan) -> Optional[ManimSceneCode]:
        """Call the code generation LLM and validate result. Returns ManimSceneCode or None."""
        class_name = re.sub(r"[^a-zA-Z0-9_]", "", scene_plan.id).title().replace("_", "")
        system_prompt = self._build_code_gen_system_prompt()
        user_message = json.dumps(scene_plan.model_dump() if hasattr(scene_plan, 'model_dump') else scene_plan.__dict__, ensure_ascii=False)

        try:
            llm_response = self._call_llm(system_prompt, user_message, temperature=self.config.temperature)
        except Exception as e:
            logger.error(f"LLM generation failed for {scene_plan.id}: {e}")
            return None

        final_code, meta = self._extract_manim_code(llm_response)
        # Save code regardless so debugging possible
        saved_path = self._save_scene_code(scene_plan.id, class_name, final_code, meta)

        # If validator rejected and returned placeholder, mark as failed
        if 'ValidationFailedScene' in final_code:
            logger.error(f"Code rejected by validator for scene {scene_plan.id}: {meta}")
            return None

        # Build ManimSceneCode container (import from your models)
        return ManimSceneCode(
            scene_id=scene_plan.id,
            scene_name=class_name,
            manim_code=final_code,
            raw_llm_output=meta.get('raw', ''),
            extraction_method='validated',
        )

    def _build_code_gen_system_prompt(self) -> str:
        # Keep the system prompt short and rely on post-validation. Encourage strict method use.
        return (
            "You are a Manim code generator. Produce working Manim Python code inside <manim>...</manim> tags."
            " Only use standard manim objects and methods. Do not invent APIs. Ensure Text(...) has font=\"sans-serif\" and MathTex uses escaped LaTeX."
        )

    # The rest of the methods (scene planning, rendering orchestration) can be
    # kept similar to your original implementation. For brevity, only key parts
    # that concern code generation and validation are implemented here.

    # Example wrapper that generates codes for many plans in parallel
    def generate_codes_for_plans(self, scene_plans: List[ScenePlan]) -> List[ManimSceneCode]:
        results: List[ManimSceneCode] = []
        with ThreadPoolExecutor(max_workers=min(len(scene_plans), 6)) as ex:
            futures = {ex.submit(self.generate_scene_code_for_plan, plan): plan for plan in scene_plans}
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    results.append(res)
        return results


# If executed directly for a smoke test, print validator behavior on sample text
    def execute(self, *args, **kwargs):
        """Concrete implementation of the abstract `execute` method required by BaseAgent.

        This implementation is intentionally lightweight and backward compatible:
        - If called with a `scene_plans` kwarg (List[ScenePlan]) it will generate codes for
          those plans and return the list of ManimSceneCode objects.
        - If called with a single `scene_plan` kwarg it will generate a single scene code.
        - Otherwise it acts as a no-op that logs the call and returns None.

        The goal is to preserve the original agent interface so your pipeline can
        instantiate ManimAgent without becoming abstract; you can later override
        this method with more specialized behavior if the pipeline expects it.
        """
        # Keep runtime imports local to avoid import cycles in some pipelines
        logger.debug(f"ManimAgent.execute called with args={args} kwargs_keys={list(kwargs.keys())}")
        # Handle explicit batch plans
        scene_plans = None
        if 'scene_plans' in kwargs:
            scene_plans = kwargs.get('scene_plans')
        elif args:
            # If first positional arg looks like a list of ScenePlan
            first = args[0]
            try:
                if isinstance(first, list):
                    scene_plans = first
            except Exception:
                scene_plans = None

        if scene_plans is not None:
            try:
                return self.generate_codes_for_plans(scene_plans)
            except Exception as e:
                logger.error(f"execute: failed to generate codes for plans: {e}")
                return None

        # Handle single plan
        if 'scene_plan' in kwargs:
            try:
                return self.generate_scene_code_for_plan(kwargs.get('scene_plan'))
            except Exception as e:
                logger.error(f"execute: failed to generate code for single plan: {e}")
                return None

        # Default fallback: nothing to do
        logger.info("ManimAgent.execute called but no scene_plans/scene_plan provided — returning None")
        return None


if __name__ == '__main__':
    sample_bad = '''```python
from manim import *

class Example(Scene):
    def construct(self):
        t = Text("Hello")
        group = VGroup()
        group.arrange(RIGHT)
        eq = MathTex(r"F_n = F_{n-1} + F_{n-2}")
        self.play(Write(t))
```
'''
    print('Extracted:')
    print(extract_python_from_response(sample_bad))
    code, meta = ManimAgent(api_key='x', model='x', output_dir=Path('.'))._extract_manim_code(sample_bad)
    print('\nValidator meta:')
    print(json.dumps(meta, indent=2, ensure_ascii=False))
