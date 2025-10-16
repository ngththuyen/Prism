import os
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pathlib import Path
from typing import Optional, Any


class Settings(BaseSettings):
    # API Keys
    google_api_key: str
    elevenlabs_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Model Selection (Gemini model IDs)
    reasoning_model: str = "gemini-2.0-flash"
    multimodal_model: str = "gemini-2.0-flash"

    # TTS Provider Selection
    tts_provider: str = "elevenlabs"  # Options: "elevenlabs", "openai"

    # ElevenLabs Settings (used when tts_provider="elevenlabs")
    elevenlabs_voice_id: str = "Qggl4b0xRMiqOwhPtVWT"
    elevenlabs_model_id: str = "eleven_v3"
    elevenlabs_stability: float = 0.75
    elevenlabs_similarity_boost: float = 0.75
    elevenlabs_style: float = 0.0
    elevenlabs_use_speaker_boost: bool = True

    # OpenAI Settings (used when tts_provider="openai")
    openai_voice: str = "alloy"
    openai_model: str = "tts-1"
    openai_endpoint: str = ""
    openai_response_format: str = "mp3"
    openai_speed: float = 1.0

    # Common TTS Settings
    tts_max_retries: int = 3
    tts_timeout: int = 120  # seconds

    # Paths
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

    # Manim Settings
    manim_quality: str = "p"  # Production quality (1080p60)
    manim_background_color: str = "#0f0f0f"
    manim_frame_rate: int = 60
    manim_render_timeout: int = 300  # seconds
    manim_max_retries: int = 3
    manim_max_scene_duration: float = 30.0  # seconds
    manim_total_video_duration_target: float = 120.0  # seconds

    # Reasoning Settings (Ignored for Gemini)
    # Note: For Anthropic models, use either reasoning tokens or effort (effort prioritized if both set)
    # For Gemini, these are ignored; adjust prompts if reasoning control is needed
    interpreter_reasoning_tokens: Optional[int] = 2048
    animation_reasoning_tokens: Optional[int] = 4096
    interpreter_reasoning_effort: Optional[str] = "low"
    animation_reasoning_effort: Optional[str] = "medium"

    # Animation Generation Settings
    animation_temperature: float = 0.5
    animation_max_retries_per_scene: int = 3
    animation_enable_simplification: bool = True

    # Script Generation Settings
    script_generation_temperature: float = 0.5
    script_generation_max_retries: int = 3
    script_generation_timeout: int = 180  # seconds

    # Video Settings
    video_codec: str = "libx264"
    video_preset: str = "medium"
    video_crf: int = 23  # Constant Rate Factor (0-51, lower = higher quality)
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"

    # Subtitle Settings
    subtitle_burn_in: bool = True  # Burn subtitles into video (hard subs)
    subtitle_font_size: int = 24
    subtitle_font_color: str = "white"
    subtitle_background: bool = True
    subtitle_background_opacity: float = 0.5
    subtitle_position: str = "bottom"  # Options: top, center, bottom

    # Video Composition Settings
    video_composition_max_retries: int = 3
    video_composition_timeout: int = 600  # seconds

    # LLM Settings
    llm_max_retries: int = 3
    llm_timeout: int = 120  # seconds

    # Language Settings
    target_language: str = "English"  # Supported: English, Chinese, Spanish, Vietnamese

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
        extra = "allow"  # Allow extra fields from environment

    def create_directories(self):
        """Create all output directories if they don't exist"""
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
    """Get settings instance with environment variables"""
    return Settings()


settings = get_settings()
settings.create_directories()