from core.scene import SceneState
from core.config import EMA_ALPHA_FAST, EMA_ALPHA_SLOW
from typing import Optional

class StateBuilder:
    def __init__(self):
        self.current_state = SceneState()

    def _ema(self, old_val: Optional[float], new_val: Optional[float], alpha: float) -> Optional[float]:
        """Apply Exponential Moving Average. If old_val is None, return new_val."""
        if new_val is None:
            return old_val
        if old_val is None:
            return new_val
        return (alpha * new_val) + ((1.0 - alpha) * old_val)

    def update(self, tracker_data: dict, visual_data: dict, timestamp: float) -> SceneState:
        """
        Updates the internal state with new data from tracker and visual analyzer,
        applying EMA smoothing to continuous values.
        """
        # We create a new SceneState instance holding the updated values
        
        # 1. Update basic fast-changing metrics
        sub_count = tracker_data.get("subjects_count", 0)
        face_detected = tracker_data.get("face_detected", False)
        
        # If subject is lost, we can choose to clear the values or keep them for a bit.
        # For simplicity in MVP, if subject count is 0, we drop coordinates to None.
        if sub_count == 0:
            cx = None
            cy = None
            sh = None
            sw = None
            fx = None
            fy = None
            fb = None
        else:
            cx = self._ema(self.current_state.subject_center_x, tracker_data.get("subject_center_x"), EMA_ALPHA_FAST)
            cy = self._ema(self.current_state.subject_center_y, tracker_data.get("subject_center_y"), EMA_ALPHA_FAST)
            sh = self._ema(self.current_state.subject_height_ratio, tracker_data.get("subject_height_ratio"), EMA_ALPHA_FAST)
            sw = self._ema(self.current_state.subject_width_ratio, tracker_data.get("subject_width_ratio"), EMA_ALPHA_FAST)
            fx = self._ema(self.current_state.face_center_x, tracker_data.get("face_center_x"), EMA_ALPHA_FAST)
            fy = self._ema(self.current_state.face_center_y, tracker_data.get("face_center_y"), EMA_ALPHA_FAST)
            fb = tracker_data.get("face_bbox") if face_detected else None
            
        # 2. Update slow-changing metrics
        blur = self._ema(self.current_state.blur_level, visual_data.get("blur_level"), EMA_ALPHA_SLOW)
        bright = self._ema(self.current_state.brightness, visual_data.get("brightness"), EMA_ALPHA_SLOW)
        face_bright = self._ema(self.current_state.face_brightness, visual_data.get("face_brightness"), EMA_ALPHA_SLOW)
        motion = self._ema(self.current_state.motion_level, visual_data.get("motion_level"), EMA_ALPHA_FAST)
        
        tilt = tracker_data.get("camera_tilt_degrees")
        if tilt is not None:
            tilt = self._ema(self.current_state.camera_tilt_degrees, tilt, EMA_ALPHA_SLOW)
            
        self.current_state = SceneState(
            timestamp=timestamp,
            subjects_count=sub_count,
            subject_center_x=cx,
            subject_center_y=cy,
            subject_height_ratio=sh,
            subject_width_ratio=sw,
            face_detected=face_detected,
            face_center_x=fx,
            face_center_y=fy,
            face_bbox=fb,
            camera_tilt_degrees=tilt,
            tilt_confidence=tracker_data.get("tilt_confidence", 0.0),
            brightness=bright or 0.0,
            face_brightness=face_bright,
            backlight=visual_data.get("backlight", False),
            blur_level=blur or 0.0,
            motion_level=motion or 0.0,
            pose_confidence=tracker_data.get("pose_confidence", 0.0),
            face_confidence=tracker_data.get("face_confidence", 0.0)
        )
        
        return self.current_state
