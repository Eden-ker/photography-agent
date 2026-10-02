import cv2
import mediapipe as mp
import numpy as np
import math
import os
import urllib.request
import threading
import time
from typing import Dict, Any, Tuple

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

def download_model_if_needed(url: str, path: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {url} to {path}...")
        urllib.request.urlretrieve(url, path)

class SubjectTracker:
    def __init__(self):
        # Ensure models are downloaded locally
        download_model_if_needed(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            POSE_MODEL_PATH
        )
        download_model_if_needed(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            FACE_MODEL_PATH
        )
        
        # State and synchronization for LIVE_STREAM mode
        self.latest_pose_result = None
        self.latest_face_result = None
        
        self.pose_event = threading.Event()
        self.face_event = threading.Event()
        
        def pose_callback(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.latest_pose_result = result
            self.pose_event.set()
            
        def face_callback(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.latest_face_result = result
            self.face_event.set()

        # Initialize Pose Landmarker
        pose_base_options = mp.tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=pose_callback,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
        
        # Initialize Face Landmarker
        face_base_options = mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=face_callback,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)
        
        self.start_time = time.time()

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process the frame using MediaPipe Tasks API (LIVE_STREAM mode).
        Returns a dictionary containing extracted spatial information.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp in ms
        current_time_ms = int((time.time() - self.start_time) * 1000)
        if not hasattr(self, 'last_timestamp_ms'):
            self.last_timestamp_ms = -1
        if current_time_ms <= self.last_timestamp_ms:
            current_time_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = current_time_ms
        timestamp_ms = current_time_ms
        
        # Reset events
        self.pose_event.clear()
        self.face_event.clear()
        
        # Submit frames async to the LIVE_STREAM pipeline
        self.pose_landmarker.detect_async(mp_image, timestamp_ms)
        self.face_landmarker.detect_async(mp_image, timestamp_ms)
        
        # Block briefly to get results for this frame (or closest past frame)
        # 50ms timeout ensures we don't hang if perception lags
        self.pose_event.wait(timeout=0.05)
        self.face_event.wait(timeout=0.05)
        
        results = {
            "subjects_count": 0,
            "subject_center_x": None,
            "subject_center_y": None,
            "subject_size_ratio": None,
            "pose_confidence": 0.0,
            "face_detected": False,
            "face_center_x": None,
            "face_center_y": None,
            "face_bbox": None,
            "face_confidence": 0.0,
            "camera_tilt_degrees": None,
            "tilt_confidence": 0.0
        }
        
        # 1. Process Pose
        if self.latest_pose_result and self.latest_pose_result.pose_landmarks:
            landmarks = self.latest_pose_result.pose_landmarks[0]
            
            results["subjects_count"] = 1
            
            # Visibility is optional but > 0.5 usually means reliable
            visible_landmarks = [lm for lm in landmarks if getattr(lm, 'visibility', 1.0) > 0.5]
            if visible_landmarks:
                xs = [lm.x for lm in visible_landmarks]
                ys = [lm.y for lm in visible_landmarks]
                
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                results["subject_center_x"] = (min_x + max_x) / 2.0
                results["subject_center_y"] = (min_y + max_y) / 2.0
                results["subject_size_ratio"] = (max_x - min_x) * (max_y - min_y)
                
                results["pose_confidence"] = sum(getattr(lm, 'visibility', 1.0) for lm in visible_landmarks) / len(visible_landmarks)
                
                # Shoulder landmarks (11=left, 12=right in MediaPipe Pose)
                if len(landmarks) > 12:
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    
                    if getattr(left_shoulder, 'visibility', 1.0) > 0.7 and getattr(right_shoulder, 'visibility', 1.0) > 0.7:
                        dy = right_shoulder.y - left_shoulder.y
                        dx = right_shoulder.x - left_shoulder.x
                        angle_rad = math.atan2(dy, dx)
                        angle_deg = math.degrees(angle_rad)
                        
                        if angle_deg > 90:
                            angle_deg -= 180
                        elif angle_deg < -90:
                            angle_deg += 180
                            
                        results["camera_tilt_degrees"] = angle_deg
                        results["tilt_confidence"] = min(getattr(left_shoulder, 'visibility', 1.0), getattr(right_shoulder, 'visibility', 1.0))
                    
        # 2. Process Face
        if self.latest_face_result and self.latest_face_result.face_landmarks:
            face_landmarks = self.latest_face_result.face_landmarks[0]
            
            results["face_detected"] = True
            
            xs = [lm.x for lm in face_landmarks]
            ys = [lm.y for lm in face_landmarks]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            results["face_center_x"] = (min_x + max_x) / 2.0
            results["face_center_y"] = (min_y + max_y) / 2.0
            results["face_bbox"] = (min_x, min_y, max_x - min_x, max_y - min_y)
            results["face_confidence"] = 1.0  # FaceLandmarker doesn't output overall confidence easily
            
            if results["subjects_count"] == 0:
                results["subjects_count"] = 1
                results["subject_center_x"] = results["face_center_x"]
                results["subject_center_y"] = results["face_center_y"]
                results["subject_size_ratio"] = (max_x - min_x) * (max_y - min_y)
                
        return results

    def close(self):
        self.pose_landmarker.close()
        self.face_landmarker.close()
