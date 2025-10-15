from pydantic import BaseModel, Fieldfrom pydantic import BaseModel, Fieldfrom pydantic import BaseModel, Field

from typing import List, Optional, Dict, Any, Tuple

from enum import Enumfrom typing import List, Optional, Dict, Any, Tuplefrom typing import List, Optional, Dict, Any, Tuple

import re

from enum import Enumfrom enum import Enum



class SceneAction(BaseModel):import reimport re

    """Represents a single visual action within a scene"""

    action_type: str

    element_type: str

    description: str

    target: str

    duration: floatclass SceneAction(BaseModel):

    parameters: Dict[str, Any] = Field(default_factory=dict)

    """Represents a single visual action within a scene"""



class ScenePlan(BaseModel):    action_type: strclass SceneAction(BaseModel):

    """Complete plan for a single animation scene"""

    id: str    element_type: str    """Represents a single visual action within a scene"""

    title: str

    description: str    description: str    action_type: str

    sub_concept_id: str

    actions: List[SceneAction]    target: str    element_type: str

    scene_dependencies: List[str] = Field(default_factory=list)

    duration: float    description: str



class ManimSceneCode(BaseModel):    parameters: Dict[str, Any] = Field(default_factory=dict)    target: str

    """Generated Manim code for a single scene"""

    scene_id: str    duration: float

    scene_name: str

    manim_code: str    parameters: Dict[str, Any] = Field(default_factory=dict)

    raw_llm_output: str

    extraction_method: str = "tags"class ScenePlan(BaseModel):



    """Complete plan for a single animation scene"""

class RenderResult(BaseModel):

    """Result of rendering a single scene"""    id: strclass ScenePlan(BaseModel):

    scene_id: str

    success: bool    title: str    """Complete plan for a single animation scene"""

    video_path: Optional[str] = None

    error_message: Optional[str] = None    description: str    id: str

    duration: Optional[float] = None

    resolution: Optional[Tuple[int, int]] = None    sub_concept_id: str    title: str

    render_time: Optional[float] = None

    file_size_mb: Optional[float] = None    actions: List[SceneAction]    description: str



    scene_dependencies: List[str] = Field(default_factory=list)    sub_concept_id: str

class AnimationResult(BaseModel):

    """Complete result of animation generation for a concept"""    actions: List[SceneAction]

    success: bool

    concept_id: str    scene_dependencies: List[str] = Field(default_factory=list)

    total_duration: Optional[float] = None

    scene_count: intclass ManimSceneCode(BaseModel):

    silent_animation_path: Optional[str] = None

    error_message: Optional[str] = None    """Generated Manim code for a single scene"""class ManimSceneCode(BaseModel):



    # Detailed results    scene_id: str    """Generated Manim code for a single scene"""

    scene_plan: List[ScenePlan]

    scene_codes: List[ManimSceneCode]    scene_name: str    scene_id: str

    render_results: List[RenderResult]

    manim_code: str    scene_name: str

    # Metadata

    generation_time: Optional[float] = None    raw_llm_output: str    manim_code: str

    total_render_time: Optional[float] = None

    models_used: Dict[str, str] = Field(default_factory=dict)    extraction_method: str = "tags"    raw_llm_output: str

    token_usage: Dict[str, int] = Field(default_factory=dict)

    extraction_method: str = "tags"



class AnimationConfig(BaseModel):

    """Configuration for animation generation"""

    quality: str = "1080p60"class RenderResult(BaseModel):

    background_color: str = "#0f0f0f"

    frame_rate: int = 60    """Result of rendering a single scene"""class RenderResult(BaseModel):

    max_scene_duration: float = 30.0

    total_video_duration_target: float = 120.0    scene_id: str    """Result of rendering a single scene"""



    max_retries_per_scene: int = 3    success: bool    scene_id: str

    temperature: float = 0.7

    video_path: Optional[str] = None    success: bool

    render_timeout: int = 300

    enable_simplification: bool = True    error_message: Optional[str] = None    video_path: Optional[str] = None

    simplify_on_retry: bool = True

    duration: Optional[float] = None    error_message: Optional[str] = None



class SceneTransition(BaseModel):    resolution: Optional[Tuple[int, int]] = None    duration: Optional[float] = None

    """Defines how scenes transition to each other"""

    from_scene: str    render_time: Optional[float] = None    resolution: Optional[Tuple[int, int]] = None

    to_scene: str

    transition_type: str = "fade"    file_size_mb: Optional[float] = None    render_time: Optional[float] = None

    duration: float = 0.5

    file_size_mb: Optional[float] = None



class AnimationMetadata(BaseModel):

    """Metadata for the complete animation generation process"""

    concept_name: strclass AnimationResult(BaseModel):

    timestamp: str

    version: str = "2.0"    """Complete result of animation generation for a concept"""class AnimationResult(BaseModel):



    # Generation statistics    success: bool    """Complete result of animation generation for a concept"""

    total_scenes_planned: int

    total_scenes_rendered: int    concept_id: str    success: bool

    successful_renders: int

    failed_renders: int    total_duration: Optional[float] = None    concept_id: str



    # Timing information    scene_count: int    total_duration: Optional[float] = None

    planning_time: Optional[float] = None

    code_generation_time: Optional[float] = None    silent_animation_path: Optional[str] = None    scene_count: int

    rendering_time: Optional[float] = None

    concatenation_time: Optional[float] = None    error_message: Optional[str] = None    silent_animation_path: Optional[str] = None

    total_time: Optional[float] = None

    error_message: Optional[str] = None

    # Resource usage

    total_tokens_used: int = 0    # Detailed results

    estimated_cost_usd: Optional[float] = None

    scene_plan: List[ScenePlan]    # Detailed results

    # File information

    total_video_size_mb: Optional[float] = None    scene_codes: List[ManimSceneCode]    scene_plan: List[ScenePlan]

    intermediate_files_count: int = 0
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