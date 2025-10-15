from pydantic import BaseModel, Fieldfrom pydantic import BaseModel, Field

from typing import List, Optional, Dict, Any, Tuplefrom typing import List, Optional, Dict, Any, Tuple

from enum import Enumfrom enum import Enum

import reimport re





class SceneAction(BaseModel):

    """Represents a single visual action within a scene"""

    action_type: strclass SceneAction(BaseModel):

    element_type: str    """Represents a single visual action within a scene"""

    description: str    action_type: str

    target: str    element_type: str

    duration: float    description: str

    parameters: Dict[str, Any] = Field(default_factory=dict)    target: str

    duration: float

    parameters: Dict[str, Any] = Field(default_factory=dict)

class ScenePlan(BaseModel):

    """Complete plan for a single animation scene"""

    id: strclass ScenePlan(BaseModel):

    title: str    """Complete plan for a single animation scene"""

    description: str    id: str

    sub_concept_id: str    title: str

    actions: List[SceneAction]    description: str

    scene_dependencies: List[str] = Field(default_factory=list)    sub_concept_id: str

    actions: List[SceneAction]

    scene_dependencies: List[str] = Field(default_factory=list)

class ManimSceneCode(BaseModel):

    """Generated Manim code for a single scene"""class ManimSceneCode(BaseModel):

    scene_id: str    """Generated Manim code for a single scene"""

    scene_name: str    scene_id: str

    manim_code: str    scene_name: str

    raw_llm_output: str    manim_code: str

    extraction_method: str = "tags"    raw_llm_output: str

    extraction_method: str = "tags"



class RenderResult(BaseModel):

    """Result of rendering a single scene"""class RenderResult(BaseModel):

    scene_id: str    """Result of rendering a single scene"""

    success: bool    scene_id: str

    video_path: Optional[str] = None    success: bool

    error_message: Optional[str] = None    video_path: Optional[str] = None

    duration: Optional[float] = None    error_message: Optional[str] = None

    resolution: Optional[Tuple[int, int]] = None    duration: Optional[float] = None

    render_time: Optional[float] = None    resolution: Optional[Tuple[int, int]] = None

    file_size_mb: Optional[float] = None    render_time: Optional[float] = None

    file_size_mb: Optional[float] = None



class AnimationResult(BaseModel):

    """Complete result of animation generation for a concept"""class AnimationResult(BaseModel):

    success: bool    """Complete result of animation generation for a concept"""

    concept_id: str    success: bool

    total_duration: Optional[float] = None    concept_id: str

    scene_count: int    total_duration: Optional[float] = None

    silent_animation_path: Optional[str] = None    scene_count: int

    error_message: Optional[str] = None    silent_animation_path: Optional[str] = None

    error_message: Optional[str] = None

    # Detailed results

    scene_plan: List[ScenePlan]    # Detailed results

    scene_codes: List[ManimSceneCode]    scene_plan: List[ScenePlan]

    render_results: List[RenderResult]    scene_codes: List[ManimSceneCode]

    render_results: List[RenderResult]

    # Metadata

    generation_time: Optional[float] = None    # Metadata

    total_render_time: Optional[float] = None    generation_time: Optional[float] = None

    models_used: Dict[str, str] = Field(default_factory=dict)    total_render_time: Optional[float] = None

    token_usage: Dict[str, int] = Field(default_factory=dict)    models_used: Dict[str, str] = Field(default_factory=dict)

    token_usage: Dict[str, int] = Field(default_factory=dict)



class AnimationConfig(BaseModel):

    """Configuration for animation generation"""class AnimationConfig(BaseModel):

    quality: str = "1080p60"    """Configuration for animation generation"""

    background_color: str = "#0f0f0f"    quality: str = "1080p60"

    frame_rate: int = 60    background_color: str = "#0f0f0f"

    max_scene_duration: float = 30.0    frame_rate: int = 60

    total_video_duration_target: float = 120.0    max_scene_duration: float = 30.0

    total_video_duration_target: float = 120.0

    max_retries_per_scene: int = 3

    temperature: float = 0.7    max_retries_per_scene: int = 3

    temperature: float = 0.7

    render_timeout: int = 300

    enable_simplification: bool = True    render_timeout: int = 300

    simplify_on_retry: bool = True    enable_simplification: bool = True

    simplify_on_retry: bool = True



class SceneTransition(BaseModel):

    """Defines how scenes transition to each other"""class SceneTransition(BaseModel):

    from_scene: str    """Defines how scenes transition to each other"""

    to_scene: str    from_scene: str

    transition_type: str = "fade"    to_scene: str

    duration: float = 0.5    transition_type: str = "fade"

    duration: float = 0.5



class AnimationMetadata(BaseModel):

    """Metadata for the complete animation generation process"""class AnimationMetadata(BaseModel):

    concept_name: str    """Metadata for the complete animation generation process"""

    timestamp: str    concept_name: str

    version: str = "2.0"    timestamp: str

    version: str = "2.0"

    # Generation statistics

    total_scenes_planned: int    # Generation statistics

    total_scenes_rendered: int    total_scenes_planned: int

    successful_renders: int    total_scenes_rendered: int

    failed_renders: int    successful_renders: int

    failed_renders: int

    # Timing information

    planning_time: Optional[float] = None    # Timing information

    code_generation_time: Optional[float] = None    planning_time: Optional[float] = None

    rendering_time: Optional[float] = None    code_generation_time: Optional[float] = None

    concatenation_time: Optional[float] = None    rendering_time: Optional[float] = None

    total_time: Optional[float] = None    concatenation_time: Optional[float] = None

    total_time: Optional[float] = None

    # Resource usage

    total_tokens_used: int = 0    # Resource usage

    estimated_cost_usd: Optional[float] = None    total_tokens_used: int = 0

    estimated_cost_usd: Optional[float] = None

    # File information

    total_video_size_mb: Optional[float] = None    # File information

    intermediate_files_count: int = 0    total_video_size_mb: Optional[float] = None
    intermediate_files_count: int = 0