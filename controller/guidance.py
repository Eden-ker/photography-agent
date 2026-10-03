import time
from enum import Enum
from typing import Optional
from core.scene import SceneState
from core.task import Task
from agent.dummy_agent import DummyAgent

class GuidanceState(Enum):
    NO_SUBJECT = 1
    ANALYZING = 2
    NO_ACTIVE_TASK = 3
    WAITING_FOR_AGENT = 4
    TASK_ACTIVE = 5
    COMPLETED = 6
    COOLDOWN = 7

class GuidanceController:
    """
    Manages the guidance state machine and task lifecycles.
    """
    def __init__(self):
        self.state = GuidanceState.NO_SUBJECT
        self.active_task: Optional[Task] = None
        self.agent = DummyAgent()
        
        self.condition_met_frames = 0
        self.REQUIRED_STABLE_FRAMES = 5
        self.cooldown_start_time = 0
        self.COOLDOWN_DURATION = 2.0
        
        self.analyzing_frames = 0
        self.REQUIRED_ANALYSIS_FRAMES = 5
        
    def evaluate(self, scene: SceneState):
        # 1. NO_SUBJECT
        if scene.subjects_count == 0:
            self.state = GuidanceState.NO_SUBJECT
            self.active_task = None
            self.analyzing_frames = 0
            self.condition_met_frames = 0
            return
            
        if self.state == GuidanceState.NO_SUBJECT:
            self.state = GuidanceState.ANALYZING
            self.analyzing_frames = 0
            
        # 2. ANALYZING
        elif self.state == GuidanceState.ANALYZING:
            self.analyzing_frames += 1
            if self.analyzing_frames >= self.REQUIRED_ANALYSIS_FRAMES:
                self.state = GuidanceState.NO_ACTIVE_TASK
                
        # 3. NO_ACTIVE_TASK
        elif self.state == GuidanceState.NO_ACTIVE_TASK:
            self.state = GuidanceState.WAITING_FOR_AGENT
            
        # 4. WAITING_FOR_AGENT
        elif self.state == GuidanceState.WAITING_FOR_AGENT:
            # Synchronous call for Stage 2 (Dummy Agent)
            task = self.agent.analyze(scene)
            if task:
                self.active_task = task
                self.state = GuidanceState.TASK_ACTIVE
                self.condition_met_frames = 0
            else:
                self.state = GuidanceState.NO_ACTIVE_TASK
                
        # 5. TASK_ACTIVE
        elif self.state == GuidanceState.TASK_ACTIVE and self.active_task:
            if self._is_condition_met(scene, self.active_task.completion_condition):
                self.condition_met_frames += 1
                if self.condition_met_frames >= self.REQUIRED_STABLE_FRAMES:
                    self.state = GuidanceState.COMPLETED
            else:
                self.condition_met_frames = 0
                
        # 6. COMPLETED
        elif self.state == GuidanceState.COMPLETED:
            self.cooldown_start_time = time.time()
            self.state = GuidanceState.COOLDOWN
            
        # 7. COOLDOWN
        elif self.state == GuidanceState.COOLDOWN:
            # Keep task displayed as "Completed" for a moment, or clear it
            if time.time() - self.cooldown_start_time >= self.COOLDOWN_DURATION:
                self.active_task = None
                self.state = GuidanceState.NO_ACTIVE_TASK
                
    def _is_condition_met(self, scene: SceneState, condition) -> bool:
        if not condition:
            return True
            
        val = getattr(scene, condition.field, None)
        if val is None:
            return False
            
        if condition.operator == "in_range":
            return abs(val - condition.target) <= condition.tolerance
        elif condition.operator == "<":
            return val < condition.target
        elif condition.operator == ">":
            return val > condition.target
        return False
