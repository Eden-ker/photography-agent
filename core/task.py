from pydantic import BaseModel, Field
from typing import Optional

class Range(BaseModel):
    min_val: float = Field(..., description="Minimum acceptable value")
    max_val: float = Field(..., description="Maximum acceptable value")

class CompositionTarget(BaseModel):
    reasoning: str = Field(..., description="Brief photographic reasoning for why this composition was chosen (e.g., 'Subject is facing right, so placing them on the left third leaves looking room').")
    subject_center_x: Range = Field(..., description="Target range for subject center X coordinate (0.0 to 1.0, e.g., 0.33 for left third)")
    face_center_y: Range = Field(..., description="Target range for face/eyes Y coordinate (0.0 to 1.0, e.g., 0.33 for top third)")
    subject_height_ratio: Range = Field(..., description="Target range for subject bounding box height relative to frame (0.0 to 1.0)")
    message: str = Field(..., description="User-facing message explaining the composition")
