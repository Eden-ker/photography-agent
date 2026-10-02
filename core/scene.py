from pydantic import BaseModel, Field
from typing import Optional

class SceneState(BaseModel):
    """
    Represents the structured state of the current camera scene.
    Values like coordinates are normalized [0.0, 1.0].
    """
    timestamp: float = Field(default=0.0, description="Time the frame was captured")
    
    # Subject tracking
    subjects_count: int = Field(default=0, description="Number of subjects detected")
    subject_center_x: Optional[float] = Field(default=None, description="Normalized X coordinate of the subject center")
    subject_center_y: Optional[float] = Field(default=None, description="Normalized Y coordinate of the subject center")
    subject_size_ratio: Optional[float] = Field(default=None, description="Ratio of the subject bounding box area to frame area")
    
    # Face tracking
    face_detected: bool = Field(default=False, description="Whether a face was clearly detected")
    face_center_x: Optional[float] = Field(default=None, description="Normalized X coordinate of the face")
    face_center_y: Optional[float] = Field(default=None, description="Normalized Y coordinate of the face")
    
    # Scene Metrics
    camera_tilt_degrees: Optional[float] = Field(default=None, description="Estimated camera tilt (horizon) in degrees")
    tilt_confidence: float = Field(default=0.0, description="Confidence in the camera tilt estimation [0, 1]")
    
    brightness: float = Field(default=0.0, description="Overall scene brightness [0, 255]")
    face_brightness: Optional[float] = Field(default=None, description="Brightness of the face region [0, 255]")
    backlight: bool = Field(default=False, description="Whether the scene has a strong backlight effect")
    
    blur_level: float = Field(default=0.0, description="Laplacian variance (lower = blurrier)")
    motion_level: float = Field(default=0.0, description="Amount of motion between frames (e.g. pixel difference)")
    
    # General Confidence
    pose_confidence: float = Field(default=0.0, description="Confidence of the overall pose detection")
    face_confidence: float = Field(default=0.0, description="Confidence of the face detection")
