import time
from enum import Enum
from typing import Optional
from core.scene import SceneState
from core.task import CompositionTarget

class GuidanceState(Enum):
    NO_SUBJECT = 1
    IDLE = 2
    ANALYZING_SNAPSHOT = 3
    GUIDING = 4

class GuidanceController:
    """
    Manages the snapshot-based guidance state machine and deterministic real-time evaluation.
    """
    def __init__(self, agent_worker):
        self.state = GuidanceState.NO_SUBJECT
        self.target: Optional[CompositionTarget] = None
        self.active_instruction: Optional[str] = None
        self.worker = agent_worker
        self.HYSTERESIS = 0.06  # Wider comfort zone to prevent oscillation
        self.MIN_HEADROOM = 0.05  # minimum normalized distance from top of frame to face
        
    def trigger_analysis(self, frame, scene: SceneState):
        """Called when user presses the Analyze button"""
        if self.worker.request_analysis(frame, scene):
            self.state = GuidanceState.ANALYZING_SNAPSHOT
            self.active_instruction = "Analyzing..."
            self.target = None # Clear stale target immediately
            
    def evaluate(self, scene: SceneState):
        # 1. NO_SUBJECT
        if scene.subjects_count == 0:
            self.state = GuidanceState.NO_SUBJECT
            self.active_instruction = None
            return
            
        if self.state == GuidanceState.NO_SUBJECT:
            if self.target:
                self.state = GuidanceState.GUIDING
            else:
                self.state = GuidanceState.IDLE
            
        # 2. IDLE
        if self.state == GuidanceState.IDLE:
            # Don't overwrite failure messages immediately
            if not self.active_instruction or "failed" not in self.active_instruction.lower():
                self.active_instruction = "Press SPACE to Analyze"
            
        # 3. ANALYZING_SNAPSHOT
        if self.state == GuidanceState.ANALYZING_SNAPSHOT:
            is_ready, result = self.worker.get_result()
            if is_ready:
                if result:
                    self.target = result
                    self.state = GuidanceState.GUIDING
                else:
                    # Failed analysis
                    self.state = GuidanceState.IDLE
                    self.active_instruction = "Analysis failed - press SPACE to retry"
                    
        # 4. GUIDING (Deterministic rules against the Target Ranges)
        if self.state == GuidanceState.GUIDING:
            if not self.target:
                self.state = GuidanceState.IDLE
                return
                
            self.active_instruction = self._compute_deterministic_instruction(scene)
            
    def _compute_deterministic_instruction(self, scene: SceneState) -> str:
        t = self.target
        # We use a hysteresis buffer so that if they are inside the box, 
        # moving back doesn't immediately cause a flicker.
        h = self.HYSTERESIS
        
        # Check X
        if scene.subject_center_x is not None:
            if scene.subject_center_x < (t.subject_center_x.min_val - h):
                return "MOVE_RIGHT"
            elif scene.subject_center_x > (t.subject_center_x.max_val + h):
                return "MOVE_LEFT"
            
        # Check Size (Height Ratio for distance)
        if scene.subject_height_ratio is not None:
            if scene.subject_height_ratio > (t.subject_height_ratio.max_val + h):
                return "MOVE_FARTHER"
            elif scene.subject_height_ratio < (t.subject_height_ratio.min_val - h):
                # Headroom Safety Check
                # Do NOT tell the user to move closer if their head is already near the top edge
                if scene.face_detected and scene.face_bbox is not None:
                    _, f_min_y, _, _ = scene.face_bbox
                    if f_min_y < self.MIN_HEADROOM:
                        # Safety override: reject moving closer because it would crop the head
                        pass
                    else:
                        return "MOVE_CLOSER"
                else:
                    # If no face is detected, fallback to using body top edge
                    body_min_y = scene.subject_center_y - (scene.subject_height_ratio / 2.0)
                    if body_min_y < self.MIN_HEADROOM:
                        pass
                    else:
                        return "MOVE_CLOSER"
            
        # Check Y (Camera tilt or user height)
        current_y = None
        if scene.face_detected and scene.face_center_y is not None:
            current_y = scene.face_center_y
        elif scene.subject_center_y is not None and scene.subject_height_ratio is not None:
            # Fallback if face not detected: assume face is in the top 15% of the body bounding box
            current_y = (scene.subject_center_y - (scene.subject_height_ratio / 2.0)) + (scene.subject_height_ratio * 0.15)
            
        if current_y is not None:
            if current_y < (t.face_center_y.min_val - h):
                return "MOVE_DOWN"
            elif current_y > (t.face_center_y.max_val + h):
                return "MOVE_UP"
            
        return "PERFECT"
