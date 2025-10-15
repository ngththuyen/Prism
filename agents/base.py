from abc import ABC, abstractmethod
import requests
import logging
import time
import json
import re
from typing import Optional, Dict, Any

# Import Google's Generative AI library
try:
    import google.generativeai as genai
    _HAS_GOOGLE_GENAI = True
except Exception:
    genai = None
    _HAS_GOOGLE_GENAI = False


class BaseAgent(ABC):
    """Base class for all AI agents using OpenRouter"""

    def __init__(self, api_key: str, base_url: str, model: str, reasoning_tokens: Optional[float] = None, reasoning_effort: Optional[str] = None, use_google: Optional[bool] = None, google_api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.reasoning_tokens = reasoning_tokens
        self.reasoning_effort = reasoning_effort
        self.logger = logging.getLogger(self.__class__.__name__)
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

        # Google GenAI integration
        # use_google controls whether to attempt to use Google Gemini via SDK
        self.use_google = bool(use_google) and _HAS_GOOGLE_GENAI
        self.google_api_key = google_api_key

        # Validate configuration
        if self.use_google:
            if not self.google_api_key:
                raise ValueError("Google API key is required when use_google=True")
            if not _HAS_GOOGLE_GENAI:
                raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
            try:
                # Configure the API
                genai.configure(api_key=self.google_api_key or self.api_key)
                self.logger.info("Google GenerativeAI configured successfully")
            except Exception as e:
                self.logger.warning(f"Failed to configure Google GenAI: {e}. Falling back to HTTP OpenRouter calls")
                self.use_google = False
                self.use_google = False

def _call_llm(
    self,
    system_prompt: str,
    user_message: str,
    temperature: float = 1.0,
    max_retries: int = 3,
    json_mode: bool = False,
) -> str:
    """Call LLM with retry logic and error handling"""

    # If Google GenAI is enabled, use it
    if self.use_google:
        for attempt in range(max_retries):
            try:
                # Initialize the model with the system prompt
                model = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=system_prompt
                )
                
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    # ⭐ FIX: Increase token limit for structured outputs (concept analysis needs long JSON)
                    max_output_tokens=int(self.reasoning_tokens) if self.reasoning_tokens else 8192,
                )
                
                # ⭐ FIX: Add JSON instruction if json_mode is True
                actual_user_message = user_message
                if json_mode:
                    actual_user_message = f"{user_message}\n\nCRITICAL: You MUST return ONLY valid JSON with no markdown formatting, no code blocks (no ```json or ```), no extra text before or after the JSON object."
                    generation_config.response_mime_type = "application/json"

                # Generate content from the user message
                response = model.generate_content(
                    actual_user_message,
                    generation_config=generation_config
                )

                # ⭐ FIX: Check if response is valid
                if not response or not response.text:
                    raise Exception(f"Empty response from Gemini. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'}")

                # Extract text content from response
                content = response.text
                
                # ⭐ FIX: Clean markdown artifacts if json_mode is True
                if json_mode:
                    # Remove markdown code blocks
                    content = re.sub(r'^```json\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
                    content = re.sub(r'^```\s*
                
                # Rough token tracking (approximate)
                try:
                    prompt_tokens = len(system_prompt.split()) + len(user_message.split())
                    completion_tokens = len(content.split())
                    self.prompt_tokens += prompt_tokens
                    self.completion_tokens += completion_tokens
                    self.total_tokens += prompt_tokens + completion_tokens
                except Exception as e:
                    self.logger.warning(f"Failed to track tokens: {e}")

                self.logger.info(f"Google GenAI LLM call successful (attempt {attempt + 1})")
                return content

            except Exception as e:
                self.logger.warning(f"Google GenAI call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise Exception(f"Google GenAI calls failed after {max_retries} attempts. Last error: {e}")

    # OpenRouter fallback (existing code remains the same)
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }

            if self.reasoning_tokens is not None:
                payload["reasoning"] = {"max_tokens": int(self.reasoning_tokens), "enabled": True}
                
            if self.reasoning_effort is not None:
                payload["reasoning"] = {"effort": self.reasoning_effort, "enabled": True}

            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            url = f"{self.base_url}/chat/completions"
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            response_data = response.json()

            if "usage" in response_data:
                self.prompt_tokens += response_data["usage"].get("prompt_tokens", 0)
                self.completion_tokens += response_data["usage"].get("completion_tokens", 0)
                self.total_tokens += response_data["usage"].get("total_tokens", 0)

            content = response_data["choices"][0]["message"]["content"]
            self.logger.info(f"LLM call successful (attempt {attempt + 1})")

            return content

        except Exception as e:
            self.logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}"
            )

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                time.sleep(wait_time)
            else:
                raise Exception(
                    f"LLM call failed after {max_retries} attempts: {e}"
                )
, '', content, flags=re.MULTILINE)
                    content = content.strip()
                    
                    # Log cleaned content for debugging
                    self.logger.debug(f"Cleaned JSON content (first 200 chars): {content[:200]}")
                    
                    # Validate it's actual JSON
                    try:
                        import json
                        json.loads(content)  # Test parse
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON validation failed: {e}")
                        self.logger.error(f"Problematic content: {content[:500]}")
                        raise Exception(f"Invalid JSON from Gemini: {e}")
                
                # Rough token tracking (approximate)
                try:
                    prompt_tokens = len(system_prompt.split()) + len(user_message.split())
                    completion_tokens = len(content.split())
                    self.prompt_tokens += prompt_tokens
                    self.completion_tokens += completion_tokens
                    self.total_tokens += prompt_tokens + completion_tokens
                except Exception as e:
                    self.logger.warning(f"Failed to track tokens: {e}")

                self.logger.info(f"Google GenAI LLM call successful (attempt {attempt + 1})")
                return content

            except Exception as e:
                self.logger.warning(f"Google GenAI call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise Exception(f"Google GenAI calls failed after {max_retries} attempts. Last error: {e}")

    # OpenRouter fallback (existing code remains the same)
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }

            if self.reasoning_tokens is not None:
                payload["reasoning"] = {"max_tokens": int(self.reasoning_tokens), "enabled": True}
                
            if self.reasoning_effort is not None:
                payload["reasoning"] = {"effort": self.reasoning_effort, "enabled": True}

            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            url = f"{self.base_url}/chat/completions"
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            response_data = response.json()

            if "usage" in response_data:
                self.prompt_tokens += response_data["usage"].get("prompt_tokens", 0)
                self.completion_tokens += response_data["usage"].get("completion_tokens", 0)
                self.total_tokens += response_data["usage"].get("total_tokens", 0)

            content = response_data["choices"][0]["message"]["content"]
            self.logger.info(f"LLM call successful (attempt {attempt + 1})")

            return content

        except Exception as e:
            self.logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}"
            )

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                time.sleep(wait_time)
            else:
                raise Exception(
                    f"LLM call failed after {max_retries} attempts: {e}"
                )

def _call_llm_structured(
    self,
    system_prompt: str,
    user_message: str,
    temperature: float = 1.0,
    max_retries: int = 3,
) -> Dict[Any, Any]:
    """Call LLM and return parsed JSON response with enhanced error handling"""

    for attempt in range(max_retries):
        try:
            response = self._call_llm(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_retries=max_retries,
                json_mode=True,
            )

            # Sanitize JSON output before parsing
            sanitized_response = self._sanitize_json_output(response)
            
            # ⭐ ADD: Additional cleaning for common issues
            # Remove any trailing incomplete strings
            sanitized_response = sanitized_response.strip()
            
            # Try to fix common JSON issues
            if not sanitized_response.endswith('}') and not sanitized_response.endswith(']'):
                self.logger.warning("JSON appears incomplete, attempting to complete it")
                # Count braces
                open_braces = sanitized_response.count('{')
                close_braces = sanitized_response.count('}')
                open_brackets = sanitized_response.count('[')
                close_brackets = sanitized_response.count(']')
                
                # Add missing closing braces
                sanitized_response += '}' * (open_braces - close_braces)
                sanitized_response += ']' * (open_brackets - close_brackets)
            
            # Attempt to parse
            try:
                parsed = json.loads(sanitized_response)
                return parsed
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parse error on attempt {attempt + 1}: {e}")
                self.logger.error(f"Problematic JSON (first 500 chars): {sanitized_response[:500]}")
                self.logger.error(f"Problematic JSON (last 200 chars): {sanitized_response[-200:]}")
                
                # If this is not the last attempt, retry
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying with increased max_output_tokens...")
                    # Increase token limit for retry
                    if self.use_google and self.reasoning_tokens:
                        old_tokens = self.reasoning_tokens
                        self.reasoning_tokens = int(self.reasoning_tokens * 1.5)
                        self.logger.info(f"Increased tokens from {old_tokens} to {self.reasoning_tokens}")
                    continue
                else:
                    raise ValueError(f"Invalid JSON response from LLM after {max_retries} attempts: {str(e)[:200]}")
        
        except Exception as e:
            if attempt < max_retries - 1:
                self.logger.warning(f"Structured call failed (attempt {attempt + 1}), retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                raise

    raise ValueError(f"Failed to get valid JSON response after {max_retries} attempts")

    def get_token_usage(self) -> Dict[str, int]:
        """Return token usage statistics"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

def _sanitize_json_output(self, text: str) -> str:
    """Remove ```json ``` code blocks and other markdown artifacts from LLM output"""
    # Remove ```json``` blocks
    sanitized = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    sanitized = re.sub(r'```\s*$', '', sanitized, flags=re.MULTILINE)
    sanitized = re.sub(r'^```\s*', '', sanitized, flags=re.MULTILINE)
    
    # Remove any leading/trailing whitespace
    sanitized = sanitized.strip()
    
    # ⭐ ADD: Remove any text before the first { or [
    first_brace = sanitized.find('{')
    first_bracket = sanitized.find('[')
    
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        sanitized = sanitized[first_brace:]
    elif first_bracket != -1:
        sanitized = sanitized[first_bracket:]
    
    return sanitized

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the agent's main task - to be implemented by subclasses"""
        pass
