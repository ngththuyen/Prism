from pydantic import BaseModel, Field
from typing import List, Optional
from agents.base import BaseAgent
import re
import json


class SubConcept(BaseModel):
    id: str
    title: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    key_points: List[str]


class ConceptAnalysis(BaseModel):
    main_concept: str
    sub_concepts: List[SubConcept]


class ConceptInterpreterAgent(BaseAgent):
    def __init__(self, api_key: str, base_url: str, model: str, reasoning_tokens: Optional[float] = None, reasoning_effort: Optional[str] = None, use_google: Optional[bool] = None, google_api_key: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url, model=model, reasoning_tokens=reasoning_tokens, reasoning_effort=reasoning_effort, use_google=use_google, google_api_key=google_api_key)

    SYSTEM_PROMPT = """
You are the Concept Interpreter Agent in an AI-powered STEM animation generation pipeline.

PROJECT CONTEXT
You are the first step in a system that transforms STEM concepts into short, clear educational videos. Your output will be consumed by:
1) A Manim Agent (to create mathematical animations),
2) A Script Generator (to write narration),
3) An Audio Synthesizer (to generate speech), and
4) A Video Compositor (to assemble the final video).

YOUR ROLE
Analyze exactly the STEM concept requested by the user and produce a structured, animation-ready breakdown that is simple, concrete, and visually actionable in the specified target language ({target_language}).

SCOPE & CLARITY RULES (Very Important)
- Focus only on the concept asked. Do not introduce variants or closely related topics unless strictly required for understanding.
- Prefer plain language and short sentences. Avoid jargon when a simple term works.
- Use examples that are easy to picture and compute (small numbers, common shapes, everyday contexts).
- Each item must be showable on screen (diagrams, steps, equations, arrows, highlights, transformations).
- Keep the sequence tight: from basics → build-up → main result → quick checks.
- If target_language is "Vietnamese", return all text (main_concept, titles, descriptions, key_points) in Vietnamese with proper accents and natural language.
- Return ONLY valid JSON with no extra text, backticks, or markdown formatting.

ANALYSIS GUIDELINES

1) Concept Decomposition (3–8 sub-concepts)
   - Start with the most concrete foundation.
   - Build step-by-step to the main idea or result.
   - Every sub-concept must be visually representable in Manim or simple diagrams.
   - Show clear dependencies (which parts must appear before others).

2) Detailed Descriptions (per sub-concept)
   - Title: 2–6 words, specific and visual, in {target_language}.
   - Description: 3–5 short sentences that explain:
     * What it is and why it matters for the main concept.
     * How it connects to the previous/next step.
     * How to show it on screen (shapes, axes, arrows, labels, motion).
     * The key “aha” insight in simple terms.
     * Use {target_language} with natural phrasing and correct accents if Vietnamese.
   - Dependencies: List of sub-concept IDs that must be shown first.

3) Key Points (4–6 per sub-concept)
   - Concrete, testable facts or relationships (numbers, formulas, directions, conditions).
   - Each should imply a visual (e.g., “draw …”, “animate …”, “label …”, “arrow from … to …”).
   - Include the minimal math/notation needed (no extra symbols).
   - Capture the “click” moment (e.g., “doubling the radius quadruples the area”).
   - Use {target_language} with proper grammar and accents if Vietnamese.

4) Pedagogical Flow
   - Concrete → abstract; simple → complex.
   - Use small, clean examples (e.g., triangles with 3–4–5; vectors with (1, 2); grids up to 5×5).
   - Include quick checkpoints (one-liners that a viewer could mentally verify).
   - Use brief, intuitive metaphors only if they directly aid the main concept (no tangents).

5) Animation-Friendly Structure
   - Specify what appears, where it appears (left/right/top), and how it moves or transforms.
   - Mention essential labels, colors (optional), and timing hints (e.g., “pause 1s after reveal”).
   - Prefer consistent notation and positions across steps.
   - If equations evolve, show term-by-term transformations (highlight moving parts).

OUTPUT FORMAT (Strict)
Return ONLY valid JSON matching exactly this structure (no extra text, no backticks):
{
  "main_concept": "string",
  "sub_concepts": [
    {
      "id": "string",
      "title": "string",
      "description": "string",
      "dependencies": ["string"],
      "key_points": ["string"]
    }
  ]
}

EXAMPLE (Easy & Clear) for “Area of a Circle” in Vietnamese:
{
  "main_concept": "Diện tích hình tròn",
  "sub_concepts": [
    {
      "id": "circle_basics",
      "title": "Hình tròn và Bán kính",
      "description": "Giới thiệu hình tròn với tâm O và bán kính r. Hiển thị bán kính là đoạn thẳng từ O đến mép hình tròn. Giải thích rằng mọi điểm trên đường tròn cách tâm O đúng r đơn vị. Đây là phép đo duy nhất cần cho diện tích.",
      "dependencies": [],
      "key_points": [
        "Vẽ hình tròn với tâm O và bán kính r",
        "Tạo hiệu ứng cho bán kính r từ O đến mép",
        "Gắn nhãn O, r và chu vi",
        "Kiểm tra: mọi điểm trên đường tròn cách O đúng r"
      ]
    },
    {
      "id": "cut_and_unroll",
      "title": "Cắt và Sắp xếp lại",
      "description": "Cắt hình tròn thành nhiều múi bằng nhau như lát pizza. Sắp xếp lại các múi xen kẽ lên xuống để tạo thành hình gần giống chữ nhật. Điều này giúp hình dung diện tích bằng cách biến hình cong thành hình đơn giản hơn.",
      "dependencies": ["circle_basics"],
      "key_points": [
        "Cắt hình tròn thành N múi (N lớn, ví dụ 16)",
        "Sắp xếp xen kẽ các múi thành hình chữ nhật zíc zắc",
        "Chiều dài trên/dưới xấp xỉ bằng nửa chu vi",
        "Chiều cao bằng bán kính r"
      ]
    },
    {
      "id": "rectangle_link",
      "title": "Gần giống Hình chữ nhật",
      "description": "Liên kết hình sắp xếp lại với hình chữ nhật có chiều cao r và chiều rộng khoảng nửa chu vi. Khi số múi tăng, các cạnh trở nên phẳng hơn. Điều này giúp tính diện tích dễ dàng hơn.",
      "dependencies": ["cut_and_unroll"],
      "key_points": [
        "Chu vi là 2πr (dùng làm chiều dài tổng)",
        "Nửa chu vi là πr (chiều rộng hình chữ nhật)",
        "Chiều cao hình chữ nhật là r",
        "Diện tích xấp xỉ là chiều rộng × chiều cao = πr × r"
      ]
    },
    {
      "id": "final_formula",
      "title": "Công thức Diện tích",
      "description": "Lấy giới hạn khi số múi tăng lên. Hình sắp xếp lại trở thành hình chữ nhật thực sự. Điều này cho công thức diện tích chính xác A = πr².",
      "dependencies": ["rectangle_link"],
      "key_points": [
        "Diện tích = (πr) × r",
        "Vậy A = πr²",
        "Làm nổi bật r² để cho thấy diện tích tăng theo bình phương bán kính",
        "Kiểm tra: gấp đôi r thì diện tích tăng gấp 4"
      ]
    }
  ]
}
"""

    def execute(self, concept: str, target_language: str = "English") -> ConceptAnalysis:
        """
        Analyze a STEM concept and return structured breakdown

        Args:
            concept: Raw text description of STEM concept (e.g., "Explain Bayes' Theorem")
            target_language: Language for output text (English, Chinese, Spanish, Vietnamese)

        Returns:
            ConceptAnalysis object with structured breakdown

        Raises:
            ValueError: If concept is invalid or LLM returns invalid response
        """

        # Input validation
        concept = concept.strip()
        if not concept:
            raise ValueError("Concept cannot be empty")
        if len(concept) > 500:
            raise ValueError("Concept description too long (max 500 characters)")

        # Sanitize input
        concept = self._sanitize_input(concept)

        self.logger.info(f"Analyzing concept: {concept} in language: {target_language}")

        # Call LLM with structured output
        user_message = f"Analyze this STEM concept and provide a structured breakdown in {target_language}:\n\n{concept}\n\nReturn ONLY valid JSON matching the exact structure provided, with no extra text, backticks, or markdown formatting."

        try:
            # Format system prompt with target_language
            system_prompt = self.SYSTEM_PROMPT.format(target_language=target_language)

            for attempt in range(4):  # Increased to 4 retries
                try:
                    response = self._call_llm_structured(
                        system_prompt=system_prompt,
                        user_message=user_message,
                        temperature=0.5,
                        max_retries=3,
                    )

                    # Log raw response for debugging
                    self.logger.debug(f"Raw LLM response (attempt {attempt + 1}): {json.dumps(response, ensure_ascii=False)}")

                    # Parse and validate with Pydantic
                    analysis = ConceptAnalysis(**response)
                    self.logger.info(f"Successfully analyzed concept: {analysis.main_concept}")
                    self.logger.info(f"Generated {len(analysis.sub_concepts)} sub-concepts")
                    return analysis

                except Exception as parse_error:
                    self.logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {parse_error}")
                    if attempt < 3:
                        self.logger.info("Retrying with stricter prompt")
                        user_message += "\nReturn ONLY valid JSON matching the exact structure provided, with no extra text, backticks, or markdown. Ensure all strings are properly closed and the JSON is complete."
                    else:
                        raise ValueError(f"Failed to parse LLM response after 4 attempts: {parse_error}")

        except Exception as e:
            self.logger.error(f"Failed to analyze concept: {e}")
            raise ValueError(f"Concept interpretation failed: {e}")

    def _sanitize_input(self, text: str) -> str:
        """Remove potentially harmful characters from input"""
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        return sanitized