from core.scene import SceneState
from core.task import Task, CompletionCondition
from core.config import GUIDANCE
from typing import Optional

class DummyAgent:
    """
    A deterministic agent that returns tasks based on hardcoded rules.
    Acts as a placeholder for the future LLM agent.
    """
    def __init__(self):
        pass
        
    def analyze(self, state: SceneState) -> Optional[Task]:
        if state.subjects_count == 0:
            return None
            
        x = state.subject_center_x
        if x is not None:
            if x > GUIDANCE["center_x_max"]:
                return Task(
                    action="MOVE_LEFT",
                    reason="Subject is too far right.",
                    completion_condition=CompletionCondition(
                        field="subject_center_x",
                        operator="in_range",
                        target=0.5,
                        tolerance=0.1
                    )
                )
            elif x < GUIDANCE["center_x_min"]:
                return Task(
                    action="MOVE_RIGHT",
                    reason="Subject is too far left.",
                    completion_condition=CompletionCondition(
                        field="subject_center_x",
                        operator="in_range",
                        target=0.5,
                        tolerance=0.1
                    )
                )
                
        size = state.subject_size_ratio
        if size is not None:
            if size < GUIDANCE["size_min"]:
                return Task(
                    action="MOVE_CLOSER",
                    reason="Subject is too far away.",
                    completion_condition=CompletionCondition(
                        field="subject_size_ratio",
                        operator=">",
                        target=GUIDANCE["size_min"] * 1.5,
                        tolerance=0.0
                    )
                )
            elif size > GUIDANCE["size_max"]:
                return Task(
                    action="MOVE_BACK",
                    reason="Subject is too close.",
                    completion_condition=CompletionCondition(
                        field="subject_size_ratio",
                        operator="<",
                        target=GUIDANCE["size_max"] * 0.8,
                        tolerance=0.0
                    )
                )
                
        return None
