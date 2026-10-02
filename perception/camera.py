import cv2
import numpy as np
from typing import Optional

class Camera:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {camera_index}")

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Reads a single frame from the camera.
        Returns the frame as a numpy array, or None if reading failed.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Releases the camera resource."""
        if self.cap.isOpened():
            self.cap.release()
