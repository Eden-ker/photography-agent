from pydantic import BaseModel, Field
from typing import Optional, Literal

class CompletionCondition(BaseModel):
    field: str = Field(..., description="The SceneState field to monitor (e.g., 'subject_center_x')")
    operator: Literal["in_range", "<", ">"] = Field(default="in_range", description="Comparison operator")
    target: float = Field(..., description="The target value")
    tolerance: float = Field(default=0.0, description="Acceptable +/- deviation from target (for in_range)")

class Task(BaseModel):
    action: str = Field(..., description="The action/instruction text, e.g. 'MOVE_LEFT'")
    priority: float = Field(default=1.0, description="Priority of the task [0.0, 1.0]")
    confidence: float = Field(default=1.0, description="Confidence in the recommendation [0.0, 1.0]")
    reason: str = Field(..., description="Reasoning for this recommendation")
    completion_condition: Optional[CompletionCondition] = Field(None, description="Condition to mark the task completed")
