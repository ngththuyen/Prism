from typing import Dict, Any, List, Optional
import json
import re
import logging
from dataclasses import dataclass
import google.generativeai as genai
from ..config import Config

@dataclass
class SubConcept:
    """Dataclass representing a sub-concept"""
    name: str
    definition: str
    explanation: str

@dataclass
class ConceptAnalysis:
    """Dataclass representing the complete analysis of a concept"""
    main_concept: str
    definition: str
    key_points: List[str]
    real_world_example: str
    visual_metaphor: str
    mathematical_notation: str
    level: str
    sub_concepts: List[SubConcept]

class ConceptInterpreterAgent:
    """
    Agent responsible for interpreting and analyzing STEM concepts
    using Google's Gemini AI model.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = Config()
        
        # Configure Gemini
        try:
            genai.configure(api_key=self.config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
            self.logger.info("Google GenerativeAI configured successfully")
        except Exception as e:
            self.logger.error(f"Failed to configure Google GenerativeAI: {e}")
            raise
    
    def analyze_concept(self, concept: str, language: str = "english") -> Dict[str, Any]:
        """
        Analyze a concept and return JSON structure with components needed for video.
        """
        self.logger.info(f"Analyzing concept: {concept} in language: {language}")

        prompt = self._build_prompt(concept, language)
        
        try:
            response = self.model.generate_content(prompt)
            
            # Extract and parse JSON from response
            json_str = self._extract_json_from_response(response.text)
            self.logger.info(f"Extracted JSON: {json_str}")
            
            parsed_response = json.loads(json_str)
            return parsed_response

        except Exception as e:
            self.logger.error(f"Failed to analyze concept: {e}")
            self.logger.error(f"Raw response: {getattr(response, 'text', 'No response')}")
            
            # Fallback response to keep pipeline running
            return self._create_fallback_response(concept, language)
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """
        Extract JSON from Gemini response text.
        Gemini sometimes returns responses with non-standard formatting.
        """
        if not response_text:
            return '{}'
            
        response_text = response_text.strip()
        
        # Find content in ```json ``` blocks
        json_block_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_block_match:
            return json_block_match.group(1).strip()
        
        # Find content in ``` ``` blocks
        block_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
        if block_match:
            return block_match.group(1).strip()
        
        # Find JSON object directly
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # If no JSON found, return original response (will cause parse error)
        return response_text
    
    def _create_fallback_response(self, concept: str, language: str) -> Dict[str, Any]:
        """Create fallback response when unable to parse response from Gemini"""
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
        Build prompt for Gemini - improved prompt for more stable JSON response
        """
        if language.lower() == "vietnamese":
            return f"""
            Hãy phân tích khái niệm STEM: "{concept}" và trả về kết quả dưới dạng JSON với cấu trúc sau:
            
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
                "level": "beginner"
            }}

            YÊU CẦU QUAN TRỌNG:
            1. Chỉ trả về JSON, không thêm bất kỳ nội dung giải thích nào khác
            2. Đảm bảo JSON là valid và đúng cấu trúc
            3. Sử dụng tiếng Việt rõ ràng, dễ hiểu
            4. Các điểm quan trọng nên ngắn gọn nhưng đầy đủ thông tin
            """
        else:
            return f"""
            Analyze the STEM concept: "{concept}" and return the result as a JSON with the following structure:
            
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
                "level": "beginner"
            }}

            IMPORTANT REQUIREMENTS:
            1. Return ONLY JSON, without any additional explanations
            2. Ensure the JSON is valid and follows the exact structure
            3. Use clear, understandable language
            4. Key points should be concise but informative
            """