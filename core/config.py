import os

# Camera Settings
CAMERA_INDEX = 0
TARGET_CAMERA_FPS = 30
TARGET_PERCEPTION_FPS = 15

# Scene State Smoothing
EMA_ALPHA_FAST = 0.5  # For fast-changing elements (e.g. bounding box)
EMA_ALPHA_SLOW = 0.2  # For slow-changing elements (e.g. brightness, blur)

# Perception Thresholds
BLUR_THRESHOLD = 100.0  # Laplacian variance threshold
BRIGHTNESS_LOW_THRESHOLD = 50
BRIGHTNESS_HIGH_THRESHOLD = 200

# Guidance Thresholds
GUIDANCE = {
    "center_x_min": 0.40,
    "center_x_max": 0.60,
    "size_min": 0.10,
    "size_max": 0.40
}

# LLM Agent Settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b" # common ollama tag
LLM_TIMEOUT_SECONDS = 15.0

# UI Settings
UI_FONT_SCALE = 0.6
UI_THICKNESS = 1
