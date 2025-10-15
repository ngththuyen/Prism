from abc import ABC, abstractmethod
import requests
import logging
import time
import json
import re
from typing import Optional, Dict, Any

# Optional Google GenAI SDK import (used for Gemini models)
try:
    from google import genai
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
        self.google_client = None

        # Validate configuration
        if self.use_google:
            if not self.google_api_key:
                raise ValueError("Google API key is required when use_google=True")
            if not _HAS_GOOGLE_GENAI:
                raise ImportError("google-genai package is required. Install with: pip install google-genai")
            try:
                # Initialize Google GenAI client with provided API key
                self.google_client = genai.Client(api_key=self.google_api_key or self.api_key)
                self.logger.info("Google GenAI client initialized for Gemini models")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Google GenAI client: {e}. Falling back to HTTP OpenRouter calls")
                self.google_client = None
                self.use_google = False

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 1.0,
        max_retries: int = 3,
        json_mode: bool = False,
    ) -> str:
        """Call LLM with retry logic and error handling using HTTPS API"""

        # If Google GenAI is enabled and client available, prefer it (Gemini)
        if self.use_google and self.google_client:
            for attempt in range(max_retries):
                try:
                    # Initialize model for generation
                    model = self.google_client.generative_model(model_name=self.model)

                    # Prepare the chat messages
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]

                    # Use the model to generate content
                    response = model.generate_content(
                        messages,
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": int(self.reasoning_tokens) if self.reasoning_tokens else None
                        }
                    )

                    # Extract text content from response
                    content = response.text

                    # Attempt to extract token usage if present
                    try:
                        prompt_tokens = sum(len(msg["content"].split()) for msg in messages)  # Rough estimation
                        completion_tokens = len(content.split())  # Rough estimation
                        self.prompt_tokens += prompt_tokens
                        self.completion_tokens += completion_tokens
                        self.total_tokens += prompt_tokens + completion_tokens
                    except Exception:
                        pass

                    self.logger.info(f"Google GenAI LLM call successful (attempt {attempt + 1})")
                    return content

                except Exception as e:
                    self.logger.warning(f"Google GenAI call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        raise Exception(f"Google GenAI calls failed after {max_retries} attempts. Last error: {e}")

        # Only use OpenRouter if explicitly configured (use_google=False) and API key provided
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
        """Call LLM and return parsed JSON response"""

        response = self._call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_retries=max_retries,
            json_mode=True,
        )

        # Sanitize JSON output before parsing
        sanitized_response = self._sanitize_json_output(response)

        try:
            return json.loads(sanitized_response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {response[:200]}...")

    def get_token_usage(self) -> Dict[str, int]:
        """Return token usage statistics"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def _sanitize_json_output(self, text: str) -> str:
        """Remove ```json ``` code blocks from LLM output"""
        # Remove ```json``` blocks
        sanitized = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        sanitized = re.sub(r'```\s*$', '', sanitized)
        return sanitized.strip()

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute the agent's main task - to be implemented by subclasses"""
        pass
