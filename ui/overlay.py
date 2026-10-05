import cv2
import numpy as np
from core.scene import SceneState
from core.config import UI_FONT_SCALE, UI_THICKNESS
from core.task import CompositionTarget

class DebugUI:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def _draw_wrapped_text(self, img, text, pos, font, font_scale, color, thickness, max_width):
        words = text.split()
        if not words:
            return
        
        lines = []
        current_line = words[0]
        for word in words[1:]:
            (w, h), _ = cv2.getTextSize(current_line + " " + word, font, font_scale, thickness)
            if w <= max_width:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        
        x, y = pos
        line_height = int(cv2.getTextSize("Ay", font, font_scale, thickness)[0][1] * 1.5) + 5
        for line in lines:
            cv2.putText(img, line, (x, y), font, font_scale, color, thickness)
            y += line_height

    def draw(self, frame: np.ndarray, state: SceneState, guidance_state=None, active_instruction: str=None, target: CompositionTarget=None) -> np.ndarray:
        h, w = frame.shape[:2]
        PANEL_WIDTH = 300
        
        # Create output frame with side panel
        output_frame = np.zeros((h, w + PANEL_WIDTH, 3), dtype=np.uint8)
        output_frame[0:h, 0:w] = frame.copy()
        
        # Draw status text
        if state.subjects_count == 0:
            cv2.putText(output_frame, "NO SUBJECT DETECTED", (20, 40), 
                        self.font, UI_FONT_SCALE * 1.5, (0, 0, 255), UI_THICKNESS + 1)
        else:
            if state.subject_center_x is not None and state.subject_center_y is not None:
                cx, cy = int(state.subject_center_x * w), int(state.subject_center_y * h)
                cv2.drawMarker(output_frame, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 20, UI_THICKNESS)
        
        # Display SceneState metrics
        y_offset = 30
        metrics = [
            f"Blur Level: {state.blur_level:.1f}",
            f"Brightness: {state.brightness:.1f}",
            f"Motion Level: {state.motion_level:.1f}",
        ]
        
        if state.backlight:
            metrics.append("WARNING: Backlight Detected!")
            
        for metric in metrics:
            color = (0, 255, 255) if "WARNING" in metric else (255, 255, 255)
            cv2.putText(output_frame, metric, (w - 300, y_offset), self.font, UI_FONT_SCALE, color, UI_THICKNESS)
            y_offset += 25
            
        # Draw Guidance State
        if guidance_state:
            state_name = guidance_state.name
            cv2.putText(output_frame, f"State: {state_name}", (20, h - 80), self.font, UI_FONT_SCALE, (200, 200, 200), UI_THICKNESS)
            
        # Draw Target Box if available
        if target:
            x_min = int(target.subject_center_x.min_val * w)
            x_max = int(target.subject_center_x.max_val * w)
            y_min = int(target.face_center_y.min_val * h)
            y_max = int(target.face_center_y.max_val * h)
            
            # Draw a box representing the target face area
            cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(output_frame, "FACE TARGET", (x_min, y_min - 10), self.font, 0.5, (0, 255, 0), 1)
            
            # Draw reasoning in the side panel
            cv2.putText(output_frame, "Gemini Reasoning:", (w + 10, 30), self.font, 0.6, (0, 255, 255), 1)
            self._draw_wrapped_text(output_frame, target.reasoning, (w + 10, 60), self.font, 0.5, (200, 255, 200), 1, PANEL_WIDTH - 20)
            
        # Draw Active Instruction
        if active_instruction:
            # Mirror the left/right instructions for the user because the camera feed is mirrored
            display_instruction = active_instruction
            if active_instruction == "MOVE_LEFT":
                display_instruction = "MOVE RIGHT"
            elif active_instruction == "MOVE_RIGHT":
                display_instruction = "MOVE LEFT"
                
            color = (0, 255, 0) if display_instruction == "PERFECT" else (0, 165, 255)
            cv2.putText(output_frame, display_instruction, (20, h - 30), self.font, UI_FONT_SCALE * 1.5, color, UI_THICKNESS + 2)
            
        return output_frame
