import cv2
import numpy as np
from core.scene import SceneState
from core.task import CompositionTarget
from PIL import Image, ImageDraw, ImageFont

class DebugUI:
    def __init__(self):
        # Load modern system fonts
        try:
            self.font_main = ImageFont.truetype("segoeui.ttf", 26)
            self.font_metrics = ImageFont.truetype("segoeui.ttf", 13)
            self.font_gemini = ImageFont.truetype("segoeui.ttf", 16)
            self.font_gemini_lbl = ImageFont.truetype("segoeui.ttf", 13)
            self.font_status = ImageFont.truetype("segoeui.ttf", 13)
        except IOError:
            self.font_main = ImageFont.load_default()
            self.font_metrics = ImageFont.load_default()
            self.font_gemini = ImageFont.load_default()
            self.font_gemini_lbl = ImageFont.load_default()
            self.font_status = ImageFont.load_default()
            
    def _get_wrapped_text_lines(self, text, font, max_width, draw):
        words = text.split()
        if not words: return []
        
        lines = []
        current_line = words[0]
        for word in words[1:]:
            w = draw.textlength(current_line + " " + word, font=font)
            if w <= max_width:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return lines

    def draw(self, frame: np.ndarray, state: SceneState, guidance_state=None, active_instruction: str=None, target: CompositionTarget=None) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for PIL, keeping alpha channel for overlays
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
        
        # Create an overlay layer for translucent drawing
        overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 1. Target Face Brackets (Subtle Overlay)
        if target:
            x_min = int(target.subject_center_x.min_val * w)
            x_max = int(target.subject_center_x.max_val * w)
            y_min = int(target.face_center_y.min_val * h)
            y_max = int(target.face_center_y.max_val * h)
            
            bracket_len = 15
            color = (200, 255, 200, 130) # Soft RGBA
            thickness = 1
            
            # Top-left
            draw.line([(x_min, y_min), (x_min + bracket_len, y_min)], fill=color, width=thickness)
            draw.line([(x_min, y_min), (x_min, y_min + bracket_len)], fill=color, width=thickness)
            # Top-right
            draw.line([(x_max, y_min), (x_max - bracket_len, y_min)], fill=color, width=thickness)
            draw.line([(x_max, y_min), (x_max, y_min + bracket_len)], fill=color, width=thickness)
            # Bottom-left
            draw.line([(x_min, y_max), (x_min + bracket_len, y_max)], fill=color, width=thickness)
            draw.line([(x_min, y_max), (x_min, y_max - bracket_len)], fill=color, width=thickness)
            # Bottom-right
            draw.line([(x_max, y_max), (x_max - bracket_len, y_max)], fill=color, width=thickness)
            draw.line([(x_max, y_max), (x_max, y_max - bracket_len)], fill=color, width=thickness)

        # 2. State Indicator (Top Left)
        if guidance_state:
            # Change IDLE to READY visually
            state_name = "READY" if guidance_state.name == "IDLE" else guidance_state.name
            state_text = f"Status: {state_name}"
            
            tw = draw.textlength(state_text, font=self.font_status)
            th = 14
            draw.rounded_rectangle([(10, 10), (10 + tw + 20, 10 + th + 15)], radius=6, fill=(20, 20, 22, 180), outline=(60, 60, 65, 100), width=1)
            draw.text((20, 16), state_text, font=self.font_status, fill=(200, 200, 200, 255))

        # 3. Technical Metrics (Top Right)
        metrics = [
            f"BLUR: {state.blur_level:.1f}",
            f"LUM: {state.brightness:.1f}",
            f"MOT: {state.motion_level:.1f}",
        ]
        if state.backlight:
            metrics.append("WARN: BACKLIGHT")
            
        metrics_w = 90
        metrics_h = len(metrics) * 18 + 10
        draw.rounded_rectangle([(w - metrics_w - 10, 10), (w - 10, 10 + metrics_h)], radius=6, fill=(20, 20, 22, 160), outline=(60, 60, 65, 80), width=1)
        
        y_offset = 14
        for m in metrics:
            color = (50, 200, 255, 200) if "WARN" in m else (160, 160, 160, 200)
            draw.text((w - metrics_w - 2, y_offset), m, font=self.font_metrics, fill=color)
            y_offset += 18

        # 4. Gemini Reasoning Panel (Bottom)
        panel_y_start = h
        if target and target.reasoning:
            max_w = w - 40
            lines = self._get_wrapped_text_lines(target.reasoning, self.font_gemini, max_w, draw)
            line_h = 22
            total_text_h = len(lines) * line_h
            
            panel_padding = 24
            panel_h = total_text_h + panel_padding + 24 # 24 padding + 24 top label space
            panel_y_start = h - panel_h - 12
            
            draw.rounded_rectangle([(12, panel_y_start), (w - 12, h - 12)], radius=10, fill=(24, 24, 28, 220), outline=(60, 60, 65, 120), width=1)
            
            draw.text((24, panel_y_start + 12), "Gemini", font=self.font_gemini_lbl, fill=(160, 160, 160, 255))
            
            text_y = panel_y_start + 36
            for line in lines:
                draw.text((24, text_y), line, font=self.font_gemini, fill=(230, 230, 230, 255))
                text_y += line_h

        # 5. Main Instruction (Floating Pill)
        INSTRUCTION_UI = {
            "MOVE_LEFT": ("→ Move Right", (255, 200, 50, 255)),
            "MOVE_RIGHT": ("← Move Left", (255, 200, 50, 255)),
            "MOVE_UP": ("↑ Move Up", (255, 200, 50, 255)),
            "MOVE_DOWN": ("↓ Move Down", (255, 200, 50, 255)),
            "MOVE_CLOSER": ("+ Move Closer", (255, 200, 50, 255)),
            "MOVE_FARTHER": ("- Move Farther", (255, 200, 50, 255)),
            "PERFECT": ("Perfect", (120, 220, 120, 255))
        }

        # Override default active instructions for a polished READY state
        is_ready_state = False
        if not active_instruction or "Press SPACE" in active_instruction:
            active_instruction = "Press SPACE to analyze"
            is_ready_state = True
        elif active_instruction == "Analysis failed - press SPACE to retry":
            is_ready_state = True

        text, color = INSTRUCTION_UI.get(active_instruction, (active_instruction, (220, 220, 220, 255)))
        font = self.font_main
        
        if is_ready_state:
            color = (200, 200, 200, 255)
            
        tw = draw.textlength(text, font=font)
        th = 30 # roughly the font size
        
        pill_pad_x = 32
        pill_pad_y = 12
        
        pill_y_bottom = panel_y_start - 24
        pill_y_top = pill_y_bottom - th - (pill_pad_y * 2)
        
        pill_x_center = w // 2
        pill_x_left = pill_x_center - (tw / 2) - pill_pad_x
        pill_x_right = pill_x_center + (tw / 2) + pill_pad_x
        
        if active_instruction == "PERFECT":
            bg_fill = (30, 70, 40, 220)
            outline_color = (60, 140, 80, 160)
        elif is_ready_state:
            bg_fill = (24, 24, 28, 180)
            outline_color = (60, 60, 65, 100)
        else:
            bg_fill = (24, 24, 28, 230)
            outline_color = (60, 60, 65, 160)
            
        draw.rounded_rectangle([(pill_x_left, pill_y_top), (pill_x_right, pill_y_bottom)], radius=24, fill=bg_fill, outline=outline_color, width=1)
        
        # Center text visually inside pill
        draw.text((pill_x_center - tw / 2, pill_y_top + pill_pad_y - 2), text, font=font, fill=color)
        
        # Composite overlay on original image
        pil_img = Image.alpha_composite(pil_img, overlay)
        
        # Convert back to BGR for OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
