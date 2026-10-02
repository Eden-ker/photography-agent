import cv2
import numpy as np

def calculate_blur(frame: np.ndarray) -> float:
    """
    Calculate the blur level of a frame using the variance of the Laplacian.
    Lower values indicate more blur.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_brightness(frame: np.ndarray) -> float:
    """
    Calculate the overall brightness of the frame.
    Returns a value between 0 and 255.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def calculate_region_brightness(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    """
    Calculate the brightness of a specific normalized region (x, y, w, h).
    """
    h, w = frame.shape[:2]
    nx, ny, nw, nh = bbox
    
    x1, y1 = int(nx * w), int(ny * h)
    x2, y2 = int((nx + nw) * w), int((ny + nh) * h)
    
    # Ensure coordinates are within bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def detect_backlight(frame: np.ndarray, face_bbox: tuple[float, float, float, float]) -> bool:
    """
    Roughly estimate if the subject is backlit by comparing face brightness to the overall or edge brightness.
    bbox is (nx, ny, nw, nh)
    """
    face_bright = calculate_region_brightness(frame, face_bbox)
    overall_bright = calculate_brightness(frame)
    
    # Simple heuristic: if face is significantly darker than the overall scene (which is bright)
    if overall_bright > 120 and face_bright < 80:
        return True
    return False

def calculate_motion(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate motion level between two consecutive frames using absolute difference.
    """
    if frame1 is None or frame2 is None or frame1.shape != frame2.shape:
        return 0.0
        
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    return float(np.mean(diff))
