import unittest
from core.scene import SceneState
from core.task import Task, CompletionCondition
from core.config import GUIDANCE
from controller.guidance import GuidanceController, GuidanceState
from agent.dummy_agent import DummyAgent

class TestGuidance(unittest.TestCase):
    def test_dummy_agent_rules(self):
        agent = DummyAgent()
        
        # Too far left -> MOVE_RIGHT
        state_left = SceneState(subjects_count=1, subject_center_x=0.1)
        task1 = agent.analyze(state_left)
        self.assertIsNotNone(task1)
        self.assertEqual(task1.action, "MOVE_RIGHT")
        self.assertEqual(task1.completion_condition.field, "subject_center_x")
        self.assertEqual(task1.completion_condition.operator, "in_range")
        
        # Too far right -> MOVE_LEFT
        state_right = SceneState(subjects_count=1, subject_center_x=0.9)
        task2 = agent.analyze(state_right)
        self.assertIsNotNone(task2)
        self.assertEqual(task2.action, "MOVE_LEFT")
        
        # Too small -> MOVE_CLOSER
        state_small = SceneState(subjects_count=1, subject_center_x=0.5, subject_size_ratio=0.05)
        task3 = agent.analyze(state_small)
        self.assertIsNotNone(task3)
        self.assertEqual(task3.action, "MOVE_CLOSER")
        
        # Perfect -> None
        state_perfect = SceneState(subjects_count=1, subject_center_x=0.5, subject_size_ratio=0.25)
        self.assertIsNone(agent.analyze(state_perfect))

    def test_guidance_state_machine(self):
        controller = GuidanceController()
        
        # Initial state
        self.assertEqual(controller.state, GuidanceState.NO_SUBJECT)
        
        # Add subject, should go to ANALYZING
        state = SceneState(subjects_count=1, subject_center_x=0.9)
        controller.evaluate(state)
        self.assertEqual(controller.state, GuidanceState.ANALYZING)
        
        # Step through ANALYZING
        for _ in range(controller.REQUIRED_ANALYSIS_FRAMES):
            controller.evaluate(state)
            
        # Evaluate twice more to step through NO_ACTIVE_TASK and WAITING_FOR_AGENT
        controller.evaluate(state)
        controller.evaluate(state)
        
        # Should transition to TASK_ACTIVE because subject is too far right
        self.assertEqual(controller.state, GuidanceState.TASK_ACTIVE)
        self.assertEqual(controller.active_task.action, "MOVE_LEFT")
        
        # Fix the subject position (satisfy condition)
        state_fixed = SceneState(subjects_count=1, subject_center_x=0.5)
        
        # Evaluate until stable
        for _ in range(controller.REQUIRED_STABLE_FRAMES):
            controller.evaluate(state_fixed)
            
        self.assertEqual(controller.state, GuidanceState.COMPLETED)
        
        # Next frame should be COOLDOWN
        controller.evaluate(state_fixed)
        self.assertEqual(controller.state, GuidanceState.COOLDOWN)

if __name__ == '__main__':
    unittest.main()
