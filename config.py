import os
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pathlib import Path
from typing import Optional, Any

class Settings(BaseSettings):
    openrouter_api_key: Optional[str] = None
    google_api_key: str
    elevenlabs_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    use_google_genai: bool = True
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    reasoning_model: str = "gemini-1.5-pro"  # Changed to more robust model
    multimodal_model: str = "gemini-1.5-pro"
    tts_provider: str = "elevenlabs"
    elevenlabs_voice_id: str = "vi-VN-NgocLamNeural"
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    elevenlabs_stability: float = 0.75
    elevenlabs_similarity_boost: float = 0.75
    elevenlabs_style: float = 0.0
    elevenlabs_use_speaker_boost: bool = True
    openai_voice: str = "alloy"
    openai_model: str = "tts-1"
    openai_endpoint: str = ""
    openai_response_format: str = "mp3"
    openai_speed: float = 1.0
    tts_max_retries: int = 3
    tts_timeout: int = 120
    output_dir: Path = Path("output")
    @property
    def scenes_dir(self) -> Path:
        return self.output_dir / "scenes"
    @property
    def animations_dir(self) -> Path:
        return self.output_dir / "animations"
    @property
    def audio_dir(self) -> Path:
        return self.output_dir / "audio"
    @property
    def scripts_dir(self) -> Path:
        return self.output_dir / "scripts"
    @property
    def final_dir(self) -> Path:
        return self.output_dir / "final"
    @property
    def analyses_dir(self) -> Path:
        return self.output_dir / "analyses"
    @property
    def rendering_dir(self) -> Path:
        return self.output_dir / "rendering"
    @property
    def generation_dir(self) -> Path:
        return self.output_dir / "generation"
    manim_quality: str = "p"
    manim_background_color: str = "#0f0f0f"
    manim_frame_rate: int = 60
    manim_render_timeout: int = 300
    manim_max_retries: int = 3
    manim_max_scene_duration: float = 30.0
    manim_total_video_duration_target: float = 120.0
    interpreter_reasoning_tokens: Optional[int] = 16384
    animation_reasoning_tokens: Optional[int] = 16384
    interpreter_reasoning_effort: Optional[str] = "medium"  # Increased effort
    animation_reasoning_effort: Optional[str] = "medium"
    animation_temperature: float = 0.5
    animation_max_retries_per_scene: int = 3
    animation_enable_simplification: bool = True
    script_generation_temperature: float = 0.5
    script_generation_max_retries: int = 3
    script_generation_timeout: int = 180
    tts_voice_id: str = "vi-VN-NgocLamNeural"
    tts_model_id: str = "eleven_multilingual_v2"
    tts_stability: float = 0.75
    tts_similarity_boost: float = 0.75
    tts_style: float = 0.0
    tts_use_speaker_boost: bool = True
    tts_max_retries: int = 3
    tts_timeout: int = 120
    video_codec: str = "libx264"
    video_preset: str = "medium"
    video_crf: int = 23
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"
    subtitle_burn_in: bool = True
    subtitle_font_size: int = 24
    subtitle_font_color: str = "white"
    subtitle_background: bool = True
    subtitle_background_opacity: float = 0.5
    subtitle_position: str = "bottom"
    video_composition_max_retries: int = 3
    video_composition_timeout: int = 600
    llm_max_retries: int = 3
    llm_timeout: int = 120
    target_language: str = "English"
    @validator('elevenlabs_api_key', 'openai_api_key', pre=True)
    def validate_tts_keys(cls, v, values):
        if 'elevenlabs_api_key' in values and 'openai_api_key' in values:
            elevenlabs_key = values['elevenlabs_api_key']
            openai_key = values['openai_api_key']
            if (not elevenlabs_key or elevenlabs_key.strip() == '') and (not openai_key or openai_key.strip() == ''):
                raise ValueError('At least one TTS API key (elevenlabs_api_key or openai_api_key) must be provided')
        return v
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
    def create_directories(self):
        for dir_path in [
            self.output_dir,
            self.scenes_dir,
            self.animations_dir,
            self.audio_dir,
            self.scripts_dir,
            self.final_dir,
            self.analyses_dir,
            self.rendering_dir,
            self.generation_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

def get_settings():
    return Settings()

settings = get_settings()
settings.create_directories()