import unittest
import numpy as np
from core.scene import SceneState
from core.task import CompositionTarget, Range
from controller.guidance import GuidanceController, GuidanceState

class MockAgentWorker:
    def __init__(self):
        self.mock_result = None
        self.requested = False
        
    def request_analysis(self, frame, state):
        self.requested = True
        return True
        
    def get_result(self):
        if self.mock_result:
            res = self.mock_result
            self.mock_result = None
            return True, res
        elif self.requested:
            return False, None
        return False, None

class TestGuidanceSnapshot(unittest.TestCase):
    def setUp(self):
        self.worker = MockAgentWorker()
        self.controller = GuidanceController(self.worker)
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_initial_state(self):
        self.assertEqual(self.controller.state, GuidanceState.NO_SUBJECT)
        self.assertIsNone(self.controller.active_instruction)
        
    def test_idle_when_subject_appears(self):
        state = SceneState(subjects_count=1)
        self.controller.evaluate(state)
        self.assertEqual(self.controller.state, GuidanceState.IDLE)
        self.assertEqual(self.controller.active_instruction, "Press SPACE to Analyze")

    def test_trigger_analysis(self):
        state = SceneState(subjects_count=1)
        self.controller.evaluate(state)
        
        self.controller.trigger_analysis(self.dummy_frame, state)
        self.assertEqual(self.controller.state, GuidanceState.ANALYZING_SNAPSHOT)
        self.assertEqual(self.controller.active_instruction, "Analyzing...")
        self.assertTrue(self.worker.requested)
        
    def test_guiding_deterministic_rules(self):
        state = SceneState(subjects_count=1, subject_center_x=0.2, subject_center_y=0.5, subject_height_ratio=0.4)
        self.controller.evaluate(state)
        
        self.controller.trigger_analysis(self.dummy_frame, state)
        
        self.worker.mock_result = CompositionTarget(
            reasoning="Looks good",
            subject_center_x=Range(min_val=0.4, max_val=0.6),
            face_center_y=Range(min_val=0.4, max_val=0.6),
            subject_height_ratio=Range(min_val=0.3, max_val=0.5),
            message="Looks good"
        )
        
        self.controller.evaluate(state)
        self.assertEqual(self.controller.state, GuidanceState.GUIDING)
        
        # Since subject_center_x is 0.2, which is < 0.4, it should say MOVE_RIGHT
        self.assertEqual(self.controller.active_instruction, "MOVE_RIGHT")
        
        # Update scene so x is good, but height is too small (e.g. 0.1)
        state.subject_center_x = 0.5
        state.subject_height_ratio = 0.1
        state.subject_center_y = 0.5
        self.controller.evaluate(state)
        self.assertEqual(self.controller.active_instruction, "MOVE_CLOSER")
        
        # Update scene to perfect
        state.subject_height_ratio = 0.4
        state.face_detected = True
        state.face_center_y = 0.5
        self.controller.evaluate(state)
        self.assertEqual(self.controller.active_instruction, "PERFECT")

    def test_headroom_safety_check(self):
        state = SceneState(subjects_count=1, subject_center_x=0.5, subject_center_y=0.5, subject_height_ratio=0.1)
        self.controller.evaluate(state)
        
        self.controller.trigger_analysis(self.dummy_frame, state)
        self.worker.mock_result = CompositionTarget(
            reasoning="Test",
            subject_center_x=Range(min_val=0.4, max_val=0.6),
            face_center_y=Range(min_val=0.4, max_val=0.6),
            subject_height_ratio=Range(min_val=0.5, max_val=0.7),
            message="Test"
        )
        self.controller.evaluate(state)
        
        # Ordinarily, height_ratio=0.1 < 0.5 means MOVE_CLOSER.
        # But let's simulate the face being too close to the top edge (headroom < 0.05).
        state.face_detected = True
        state.face_bbox = (0.45, 0.02, 0.1, 0.1)  # min_y is 0.02, which is < 0.05 MIN_HEADROOM
        
        self.controller.evaluate(state)
        # Because of headroom safety check, it should suppress MOVE_CLOSER and fall through to PERFECT 
        # (or whatever the next failing check is, but in this case everything else is perfect).
        self.assertNotEqual(self.controller.active_instruction, "MOVE_CLOSER")
        self.assertEqual(self.controller.active_instruction, "PERFECT")

    def test_failed_analysis_clears_target(self):
        state = SceneState(subjects_count=1)
        self.controller.evaluate(state)
        
        # Trigger analysis 1 (success)
        self.controller.trigger_analysis(self.dummy_frame, state)
        self.worker.mock_result = CompositionTarget(
            reasoning="Test",
            subject_center_x=Range(min_val=0.4, max_val=0.6),
            face_center_y=Range(min_val=0.4, max_val=0.6),
            subject_height_ratio=Range(min_val=0.5, max_val=0.7),
            message="Test"
        )
        self.controller.evaluate(state)
        self.assertEqual(self.controller.state, GuidanceState.GUIDING)
        self.assertIsNotNone(self.controller.target)
        
        # Trigger analysis 2 (failure)
        self.controller.trigger_analysis(self.dummy_frame, state)
        # Verify target is cleared immediately
        self.assertIsNone(self.controller.target)
        
        # Simulate worker returning None (failure)
        self.worker.requested = False
        self.worker.mock_result = None
        
        # To simulate the worker being done, we actually need to change how mock worker works, 
        # but setting it to return True, None is what it does when it finishes with error.
        # Let's override get_result for a failure
        self.worker.get_result = lambda: (True, None)
        
        self.controller.evaluate(state)
        self.assertEqual(self.controller.state, GuidanceState.IDLE)
        self.assertIn("failed", self.controller.active_instruction)
        
        # Evaluate again, ensure it doesn't instantly overwrite "failed" message with "Press SPACE"
        self.controller.evaluate(state)
        self.assertIn("failed", self.controller.active_instruction)

if __name__ == '__main__':
    unittest.main()
