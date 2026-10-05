import os
import cv2
import tempfile
from core.scene import SceneState
from core.task import CompositionTarget
import numpy as np

try:
    from google import genai
    from PIL import Image
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class GeminiAgent:
    def __init__(self):
        if not HAS_GENAI:
            print("[GeminiAgent] Warning: google-genai is not installed.")
            
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("[GeminiAgent] ERROR: GEMINI_API_KEY not found in environment.")
            
        self.model_name = "gemini-3.5-flash"

    def _build_prompt(self, state: SceneState) -> str:
        # Format current state for the VLM
        cx = f"{state.subject_center_x:.2f}" if state.subject_center_x is not None else "Unknown"
        cy = f"{state.subject_center_y:.2f}" if state.subject_center_y is not None else "Unknown"
        sh = f"{state.subject_height_ratio:.2f}" if state.subject_height_ratio is not None else "Unknown"
        
        face_info = "No face detected."
        if state.face_detected and state.face_bbox is not None:
            _, f_min_y, _, f_h = state.face_bbox
            f_cy = f_min_y + (f_h / 2.0)
            face_info = f"Face detected. Face Center Y: {f_cy:.2f}, Face top edge (min_y): {f_min_y:.2f}."
            
        return f"""You are a professional photography composition assistant.
Look at the entire image like a photographer and decide what composition would make this particular photo better.
Do NOT hard-code "put the person in the center". You may place the subject left/right/center and larger/smaller depending on the scene (rule of thirds, looking room, negative space, environmental context).

CURRENT SUBJECT STATE:
- Subject Center X: {cx}
- Subject Height Ratio: {sh} (0.0 to 1.0, where 1.0 means filling the entire height of the screen)
- Face Info: {face_info}

CRITICAL RULES:
1. Vertically frame using `face_center_y` (where the subject's eyes/face are). For example, 0.33 places the face on the top third line.
2. `subject_height_ratio` represents the bounding box HEIGHT relative to the full frame height.
   - A full-body shot is ~0.60 - 0.90 height.
   - A standard portrait (chest up) might be ~0.40 - 0.60 height.
   - Do NOT recommend a height ratio that forces the subject's face/head off the top of the frame.
3. If the current composition is already good, prefer small, subtle corrections rather than drastic moves.
4. Provide comfortable target ranges (e.g., a width of 0.15 for the range) rather than exact points.

Provide your final recommendation in JSON format matching this structure:
{{
  "reasoning": "Briefly explain the photographic decision made before outputting coordinates.",
  "subject_center_x": {{"min_val": 0.4, "max_val": 0.6}},
  "face_center_y": {{"min_val": 0.25, "max_val": 0.4}},
  "subject_height_ratio": {{"min_val": 0.4, "max_val": 0.6}},
  "message": "User-facing instruction explaining why to move."
}}
Only output the JSON object at the end.
"""

    def analyze(self, frame: np.ndarray, state: SceneState) -> CompositionTarget | None:
        if not HAS_GENAI or not self.api_key:
            print("[GeminiAgent] Cannot run analysis due to missing dependencies or API key.")
            return None
            
        print("[GeminiAgent] Analyzing snapshot...")
        
        prompt = self._build_prompt(state)
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            client = genai.Client()
            response = client.models.generate_content(
                model=self.model_name,
                contents=[prompt, img],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CompositionTarget,
                    temperature=0.2,
                )
            )
            
            target = CompositionTarget.model_validate_json(response.text)
            print(f"[GeminiAgent] Reasoning: {target.reasoning}")
            return target
            
        except Exception as e:
            print(f"[GeminiAgent] Inference failed: {e}")
            return None
