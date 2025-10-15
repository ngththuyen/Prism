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
- Return ONLY valid JSON with no extra text, backticks, or markdown formatting. Example: {"main_concept": "Đạo hàm", "sub_concepts": []}
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
   - Written in {target_language}.
   - Must be visually showable (e.g., "Show f(x) = x² curve" or "Highlight slope as Δy/Δx").
   - Avoid abstract claims without visual grounding.
TASK
Given the STEM concept: "{concept}", produce a JSON object matching the structure:
{
  "main_concept": "<main concept in target_language>",
  "sub_concepts": [
    {
      "id": "<unique-id>",
      "title": "<short, visual title>",
      "description": "<3–5 sentences describing the sub-concept>",
      "dependencies": ["<id of dependent sub-concept>", ...],
      "key_points": ["<point 1>", "<point 2>", ...]
    },
    ...
  ]
}
- Ensure all text fields are in {target_language} (e.g., Vietnamese with proper accents if specified).
- Ensure JSON is valid, with no extra text or markdown.
- For concept "giải thích đạo hàm" in Vietnamese, main_concept should be "Đạo hàm" or similar.
"""
    def execute(self, concept: str, target_language: str = "English") -> ConceptAnalysis:
        """Analyze a STEM concept and return structured breakdown"""
        self.logger.info(f"Analyzing concept: {concept} in language: {target_language}")
        sanitized_concept = self._sanitize_input(concept)
        user_message = f"""
Analyze the STEM concept: "{sanitized_concept}"
Target language: {target_language}
Follow the guidelines and return a JSON object matching the specified structure.
"""
        for attempt in range(4):
            try:
                self.logger.debug(f"Calling LLM for concept analysis (attempt {attempt + 1})")
                response = self._call_llm_structured(
                    system_prompt=self.SYSTEM_PROMPT.format(concept=sanitized_concept, target_language=target_language),
                    user_message=user_message,
                    temperature=0.7,
                    max_retries=1
                )
                self.logger.debug(f"Received LLM response: {json.dumps(response, ensure_ascii=False)[:1000]}")
                analysis = ConceptAnalysis(**response)
                self.logger.info("Concept analysis parsed successfully")
                return analysis
            except Exception as parse_error:
                self.logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {parse_error}")
                if attempt < 3:
                    self.logger.info("Retrying with stricter prompt")
                    user_message += "\nReturn ONLY valid JSON matching the exact structure provided, with no extra text, backticks, or markdown."
                else:
                    raise ValueError(f"Failed to parse LLM response after 4 attempts: {parse_error}")
        raise ValueError("Failed to analyze concept after all retries")
    def _call_llm_structured(self, system_prompt: str, user_message: str, temperature: float = 0.7, max_retries: int = 3) -> dict:
        """Override _call_llm_structured to clean LLM response before parsing"""
        self.logger.debug(f"Entering custom _call_llm_structured with user_message: {user_message[:500]}...")
        for attempt in range(max_retries + 1):
            try:
                response = super()._call_llm(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    temperature=temperature,
                    max_retries=1,
                    json_mode=True
                )
                self.logger.debug(f"Raw LLM response before cleaning (attempt {attempt + 1}): {response[:1000]}")
                cleaned_response = self._clean_llm_response(response)
                self.logger.debug(f"Cleaned LLM response (attempt {attempt + 1}): {cleaned_response[:1000]}")
                response_json = json.loads(cleaned_response)
                self.logger.info(f"JSON parsed successfully on attempt {attempt + 1}")
                return response_json
            except Exception as e:
                self.logger.warning(f"LLM call failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    self.logger.info("Retrying LLM call")
                    continue
                raise ValueError(f"Failed to get valid JSON response after {max_retries + 1} attempts: {e}")
    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response to ensure valid JSON"""
        response = re.sub(r'```json\n|```', '', response)
        response = re.sub(r'\n\s*\n', '\n', response.strip())
        response = response.strip()
        if not response.startswith(('{', '[')):
            response = '{' + response if 'main_concept' in response else '[' + response
        if not response.endswith(('}', ']')):
            open_braces = response.count('{')
            close_braces = response.count('}')
            open_brackets = response.count('[')
            close_brackets = response.count(']')
            response += '}' * (open_braces - close_braces)
            response += ']' * (open_brackets - close_brackets)
        return response
    def _sanitize_input(self, text: str) -> str:
        """Remove potentially harmful characters from input"""
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        return sanitized
    
    def analyze_concept(self, concept: str, language: str = "english") -> Dict[str, Any]:
        """
        Phân tích khái niệm và trả về cấu trúc JSON với các thành phần cần thiết cho video.
        """
        self.logger.info(f"Analyzing concept: {concept} in language: {language}")

        prompt = self._build_prompt(concept, language)
        response = self.model.generate_content(prompt)

        try:
            # Xử lý response để extract JSON
            json_str = self._extract_json_from_response(response.text)
            self.logger.info(f"Extracted JSON: {json_str}")
            
            parsed_response = json.loads(json_str)
            return parsed_response

        except Exception as e:
            self.logger.error(f"Failed to analyze concept: {response.text}")
            # Fallback response để pipeline có thể tiếp tục
            return self._create_fallback_response(concept, language)
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """
        Extract JSON từ response text của Gemini.
        Gemini đôi khi trả về response với format không chuẩn.
        """
        # Loại bỏ các ký tự thừa và tìm JSON object
        response_text = response_text.strip()
        
        # Tìm content trong ```json ``` blocks (nếu có)
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_block_match:
            return json_block_match.group(1).strip()
        
        # Tìm content trong ``` ``` blocks (nếu có)
        block_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
        if block_match:
            return block_match.group(1).strip()
        
        # Tìm JSON object trực tiếp
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # Nếu không tìm thấy JSON, trả về response gốc (sẽ bị lỗi parse)
        return response_text
    
    def _create_fallback_response(self, concept: str, language: str) -> Dict[str, Any]:
        """Tạo fallback response khi không thể parse được response từ Gemini"""
        if language.lower() == "vietnamese":
            return {
                "main_concept": concept,
                "definition": f"Định nghĩa cơ bản về {concept}",
                "key_points": [
                    f"Khái niệm quan trọng về {concept}",
                    f"Ứng dụng thực tế của {concept}",
                    f"Ý nghĩa của {concept} trong STEM"
                ],
                "real_world_example": f"Ví dụ thực tế minh họa cho {concept}",
                "visual_metaphor": f"Ẩn dụ trực quan cho {concept}",
                "mathematical_notation": "Công thức toán học liên quan",
                "level": "beginner"
            }
        else:
            return {
                "main_concept": concept,
                "definition": f"Basic definition of {concept}",
                "key_points": [
                    f"Key concept about {concept}",
                    f"Practical applications of {concept}",
                    f"Significance of {concept} in STEM"
                ],
                "real_world_example": f"Real-world example illustrating {concept}",
                "visual_metaphor": f"Visual metaphor for {concept}",
                "mathematical_notation": "Related mathematical formula",
                "level": "beginner"
            }

    def _build_prompt(self, concept: str, language: str) -> str:
        """
        Xây dựng prompt cho Gemini - cải thiện prompt để có response JSON ổn định hơn
        """
        if language.lower() == "vietnamese":
            return f"""
            Hãy phân tích khái niệm: "{concept}" và trả về kết quả dưới dạng JSON với cấu trúc sau:
            
            {{
                "main_concept": "Tên chính xác của khái niệm",
                "definition": "Định nghĩa ngắn gọn, dễ hiểu",
                "key_points": [
                    "Điểm quan trọng 1",
                    "Điểm quan trọng 2", 
                    "Điểm quan trọng 3"
                ],
                "real_world_example": "Ví dụ thực tế minh họa",
                "visual_metaphor": "Ẩn dụ trực quan để minh họa khái niệm",
                "mathematical_notation": "Ký hiệu toán học (nếu có)",
                "level": "Cấp độ giải thích (beginner, intermediate, advanced)"
            }}

            YÊU CẦU QUAN TRỌNG:
            1. Chỉ trả về JSON, không thêm bất kỳ nội dung giải thích nào khác
            2. Đảm bảo JSON là valid và đúng cấu trúc
            3. Sử dụng tiếng Việt rõ ràng, dễ hiểu
            4. Các điểm quan trọng nên ngắn gọn nhưng đầy đủ thông tin
            """
        else:
            return f"""
            Analyze the concept: "{concept}" and return the result as a JSON with the following structure:
            
            {{
                "main_concept": "Exact name of the concept",
                "definition": "Brief, easy-to-understand definition", 
                "key_points": [
                    "Key point 1",
                    "Key point 2",
                    "Key point 3"
                ],
                "real_world_example": "Practical example illustrating the concept",
                "visual_metaphor": "Visual metaphor to illustrate the concept",
                "mathematical_notation": "Mathematical notation (if applicable)",
                "level": "Explanation level (beginner, intermediate, advanced)"
            }}

            IMPORTANT REQUIREMENTS:
            1. Return ONLY JSON, without any additional explanations
            2. Ensure the JSON is valid and follows the exact structure
            3. Use clear, understandable language
            4. Key points should be concise but informative
            """