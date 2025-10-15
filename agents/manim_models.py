from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Tuple, Union, Literal
from enum import Enum
import re


class EasingType(str, Enum):
    """Valid easing types that map to Manim rate functions"""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    SMOOTH = "smooth"
    RUSH_INTO = "rush_into"
    RUSH_FROM = "rush_from"
    EXPONENTIAL_DECAY = "exponential_decay"


class AnimationStyle(str, Enum):
    """Animation styles available in Manim"""
    FADE = "fade"
    GROW = "grow"
    TRANSFORM = "transform"
    WRITE = "write"
    INDICATE = "indicate"
    FOCUS = "focus"


class AnimationConfig(BaseModel):
    """Configuration for a single animation"""
    easing: Optional[EasingType] = Field(default=EasingType.SMOOTH, description="The easing function to use")
    duration: Optional[float] = Field(default=None, description="Override the default duration")
    style: Optional[AnimationStyle] = Field(default=None, description="The animation style to use")
    lag_ratio: Optional[float] = Field(default=None, description="Delay between elements in group animations")
    
    @validator('easing')
    def validate_easing(cls, v):
        if v and not isinstance(v, EasingType):
            try:
                return EasingType(v)
            except ValueError:
                return EasingType.SMOOTH
        return v


class ActionParameters(BaseModel):
    """Parameters for scene actions with comprehensive animation support"""
    # Content parameters
    text: Optional[str] = None
    color: Optional[str] = None
    equation: Optional[str] = None
    style: Optional[str] = None
    label: Optional[str] = None
    
    # Target parameters
    from_target: Optional[str] = None
    to_target: Optional[str] = None
    
    # Visual parameters
    position: Optional[Union[str, List[float]]] = None
    scale: Optional[float] = None
    opacity: Optional[float] = None
    
    # Animation parameters
    animation: Optional[AnimationConfig] = Field(
        default_factory=AnimationConfig,
        description="Animation-specific configuration"
    )
    
    # Additional parameters
    spans: Optional[List[Dict[str, str]]] = None
    branches: Optional[List[Dict[str, str]]] = None

    @validator('*', pre=True)
    def remove_easing(cls, v):
        """Remove easing parameter if present in nested structures"""
        if isinstance(v, dict):
            v.pop('easing', None)
        return v


class SceneAction(BaseModel):
    """Represents a single visual action within a scene"""
    action_type: str
    element_type: str
    description: str
    target: str
    duration: float
    parameters: ActionParameters = Field(default_factory=ActionParameters)


class ScenePlan(BaseModel):
    """Complete plan for a single animation scene"""
    id: str
    title: str
    description: str
    sub_concept_id: str
    actions: List[SceneAction]
    scene_dependencies: List[str] = Field(default_factory=list)

class ManimSceneCode(BaseModel):
    """Generated Manim code for a single scene"""
    scene_id: str
    scene_name: str
    manim_code: str
    raw_llm_output: str
    extraction_method: str = "tags"


class RenderResult(BaseModel):
    """Result of rendering a single scene"""
    scene_id: str
    success: bool
    video_path: Optional[str] = None
    error_message: Optional[str] = None
    duration: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None
    render_time: Optional[float] = None
    file_size_mb: Optional[float] = None


class AnimationResult(BaseModel):
    """Complete result of animation generation for a concept"""
    success: bool
    concept_id: str
    total_duration: Optional[float] = None
    scene_count: int
    silent_animation_path: Optional[str] = None
    error_message: Optional[str] = None

    # Detailed results
    scene_plan: List[ScenePlan]
    scene_codes: List[ManimSceneCode]
    render_results: List[RenderResult]

    # Metadata
    generation_time: Optional[float] = None
    total_render_time: Optional[float] = None
    models_used: Dict[str, str] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)


class AnimationConfig(BaseModel):
    """Configuration for animation generation"""
    quality: str = "1080p60"
    background_color: str = "#0f0f0f"
    frame_rate: int = 60
    max_scene_duration: float = 30.0
    total_video_duration_target: float = 120.0

    max_retries_per_scene: int = 3
    temperature: float = 0.7

    render_timeout: int = 300
    enable_simplification: bool = True
    simplify_on_retry: bool = True


class SceneTransition(BaseModel):
    """Defines how scenes transition to each other"""
    from_scene: str
    to_scene: str
    transition_type: str = "fade"
    duration: float = 0.5


class AnimationMetadata(BaseModel):
    """Metadata for the complete animation generation process"""
    concept_name: str
    timestamp: str
    version: str = "2.0"

    # Generation statistics
    total_scenes_planned: int
    total_scenes_rendered: int
    successful_renders: int
    failed_renders: int

    # Timing information
    planning_time: Optional[float] = None
    code_generation_time: Optional[float] = None
    rendering_time: Optional[float] = None
    concatenation_time: Optional[float] = None
    total_time: Optional[float] = None

    # Resource usage
    total_tokens_used: int = 0
    estimated_cost_usd: Optional[float] = None

    # File information
    total_video_size_mb: Optional[float] = None
    intermediate_files_count: int = 0