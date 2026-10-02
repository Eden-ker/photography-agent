import cv2
import numpy as np
from core.scene import SceneState
from core.config import UI_FONT_SCALE, UI_THICKNESS

class DebugUI:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw(self, frame: np.ndarray, state: SceneState) -> np.ndarray:
        """
        Draw debug information on the frame based on the SceneState.
        """
        output_frame = frame.copy()
        h, w = output_frame.shape[:2]
        
        # Draw status text
        if state.subjects_count == 0:
            cv2.putText(output_frame, "NO SUBJECT DETECTED", (20, 40), 
                        self.font, UI_FONT_SCALE * 1.5, (0, 0, 255), UI_THICKNESS + 1)
        else:
            # Draw subject center
            if state.subject_center_x is not None and state.subject_center_y is not None:
                cx, cy = int(state.subject_center_x * w), int(state.subject_center_y * h)
                cv2.drawMarker(output_frame, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 20, UI_THICKNESS)
            
            # Draw face center
            if state.face_detected and state.face_center_x is not None and state.face_center_y is not None:
                fx, fy = int(state.face_center_x * w), int(state.face_center_y * h)
                cv2.circle(output_frame, (fx, fy), 10, (0, 255, 0), UI_THICKNESS)
        
        # Display SceneState metrics
        y_offset = 30
        metrics = [
            f"FPS (Perception target): 15",
            f"Blur Level: {state.blur_level:.1f}",
            f"Brightness: {state.brightness:.1f}",
            f"Motion Level: {state.motion_level:.1f}",
        ]
        
        if state.camera_tilt_degrees is not None and state.tilt_confidence > 0.6:
            metrics.append(f"Tilt: {state.camera_tilt_degrees:.1f} deg")
        else:
            metrics.append("Tilt: N/A (low conf)")
            
        if state.backlight:
            metrics.append("WARNING: Backlight Detected!")
            
        for metric in metrics:
            color = (0, 255, 255) if "WARNING" in metric else (255, 255, 255)
            cv2.putText(output_frame, metric, (w - 300, y_offset), self.font, UI_FONT_SCALE, color, UI_THICKNESS)
            y_offset += 25
            
        return output_frame
