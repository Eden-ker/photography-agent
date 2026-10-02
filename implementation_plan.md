# AI Photography Assistant - Implementation Plan

This document outlines the architecture, technology choices, and implementation plan for the local, real-time AI photography assistant MVP.

## A. Recommended Architecture

The system will use an asynchronous, decoupled architecture to satisfy the constraint that the LLM must not block the camera feed. 

1. **Perception Loop**: Continuously captures frames from the webcam, runs OpenCV algorithms (blur, brightness) and MediaPipe (Face/Pose), and updates a shared `SceneState`. Target frame rate: ~10-15 FPS for perception processing, while Camera/UI target ~30 FPS.
2. **Guidance Controller**: Evaluates the current `SceneState` against the active task's completion conditions. Manages task lifecycle, persistence, hysteresis, cooldown, and completion.
3. **Photography Agent (Agent Worker)**: A single background worker running on a queue (or equivalent mechanism) that processes the latest relevant reasoning request, preventing thread proliferation. Sends the `SceneState` to the local LLM and parses the JSON response.
4. **UI/Overlay**: Only presents the current instruction, drawing the latest frame and the active task recommendation/arrow on screen.

## B. Technology Choices

- **Language**: Python 3.10+
- **Computer Vision**: 
  - `opencv-python`: Camera I/O, image processing (Laplacian variance for blur, histogram/brightness calculation), and UI drawing.
  - `mediapipe`: Face and Pose tracking. Highly optimized for CPU execution.
- **LLM Runtime**: **Ollama**. We will rely on Pydantic validation and retry/fallback logic to handle any structured output anomalies, avoiding heavier strict-grammar runtimes for the MVP.
- **LLM Model Candidate**: **Qwen2.5-1.5B-Instruct** (Primary). **Qwen3.5-2B** (Future Benchmark Candidate). 
  *Hardware Note:* With 16 GB RAM and an NVIDIA MX330 (2 GB VRAM), the system will be treated as CPU-first for the LLM. We will not assume a 2GB quantized model fits cleanly into 2GB VRAM. We will benchmark actual runtime memory usage and latency before deciding whether GPU offloading provides a tangible benefit.
- **Data Validation**: `pydantic` to enforce type safety and strict schema for the `SceneState` and Agent outputs.

## C. Module and Repository Structure

```text
photography-agent/
│ core/
│   ├── config.py             # Global constants, thresholds, model settings
│   ├── scene.py              # Pydantic models for SceneState
│   └── task.py               # Pydantic models for Agent Actions/Recommendations
│ perception/
│   ├── camera.py             # Webcam capture abstraction
│   ├── visual.py             # OpenCV algorithms (blur, brightness, tilt)
│   ├── tracker.py            # MediaPipe Pose and Face extraction
│   └── state_builder.py      # Aggregates and applies moving average smoothing
│ agent/
│   ├── llm_client.py         # Ollama REST API interface with retry/fallback
│   ├── photography_agent.py  # Prompt construction and JSON parsing
│   └── worker.py             # Single background queue worker for LLM requests
│ controller/
│   └── guidance.py           # State machine, hysteresis, task lifecycle
│ ui/
│   └── overlay.py            # Visual drawing (arrows, text feedback)
├── main.py                   # Main loop and thread coordination
├── requirements.txt
└── README.md
```

## D. Data Flow

1. **Frame Capture**: `camera.py` yields a frame (~30 FPS).
2. **Perception**: At ~10-15 FPS, the frame goes to `visual.py` and `tracker.py`.
3. **State Aggregation**: `state_builder.py` creates a raw `SceneState`, then applies an Exponential Moving Average (EMA) to prevent jitter.
4. **State Machine**: `guidance.py` evaluates the smoothed `SceneState`.
   - If `NO_ACTIVE_TASK` and conditions are stable, it pushes a request to the Agent Worker queue and transitions to `WAITING_FOR_AGENT`.
   - If `TASK_ACTIVE`, it monitors the `completion_condition`.
5. **Reasoning**: The single Agent Worker pulls the latest request (discarding stale ones), queries the LLM, validates output via Pydantic, and returns a structured task.
6. **Execution**: Controller accepts the task, transitions to `TASK_ACTIVE`.
7. **Rendering**: `overlay.py` displays the instruction (e.g., "Move Left ←").

## E. Agent Input/Output Contract

### Input (Prompt Context)
The LLM will be provided with a system prompt defining its persona and allowed actions, followed by the current `SceneState` serialized to JSON:
```json
{
  "scene": {
    "subjects_count": 1,
    "subject_center_x": 0.85, 
    "subject_center_y": 0.50,
    "subject_size_ratio": 0.15,
    "blur_level": "LOW",
    "brightness": "GOOD",
    "camera_tilt_degrees": 2.0
  },
  "history": {
    "previous_tasks": ["MOVE_CLOSER", "MOVE_LEFT"]
  }
}
```
*(Note: X/Y coordinates will be normalized [0.0, 1.0].)*

### Output (LLM JSON Response)
```json
{
  "action": "MOVE_LEFT",
  "priority": 0.95,
  "confidence": 0.90,
  "reason": "The subject is positioned too far to the right of the frame.",
  "completion_condition": {
    "field": "subject_center_x",
    "target": 0.50,
    "tolerance": 0.10
  }
}
```

## F. Guidance State Machine

1. `NO_SUBJECT`: Reset state. Waiting for `subjects_count > 0`.
2. `ANALYZING`: Subject found. Wait for N frames to build a stable, non-jittery scene state.
3. `NO_ACTIVE_TASK`: State is stable. Trigger Agent Worker.
4. `WAITING_FOR_AGENT`: LLM is generating via the background worker.
5. `TASK_ACTIVE`: Instruction is displayed on screen. Controller monitors `completion_condition`.
6. `COMPLETED`: Condition met for X consecutive frames.
7. `COOLDOWN`: Wait Y seconds to prevent instruction spam.
8. -> Returns to `NO_ACTIVE_TASK`.

## G. MVP Implementation Plan

- **Stage 1: Foundation & Perception**: Setup webcam, OpenCV drawing, and MediaPipe tracking. Implement Pydantic `SceneState` and EMA smoothing logic. Target 10-15 FPS for perception.
- **Stage 2: Core Logic & State Machine**: Implement the Guidance Controller state machine and task tracking logic. Use a **Dummy Agent** (returning deterministic mock recommendations) to validate the complete perception -> scene state -> controller -> UI pipeline before introducing the LLM.
- **Stage 3: LLM Integration**: Implement the single Agent Worker queue, Ollama client, prompt templates, and Pydantic validation/retry logic. Wire the Qwen2.5-1.5B-Instruct agent into the worker.
- **Stage 4: Tuning & Benchmarking**: Refine the visual overlay (arrows, text). Tune hysteresis and cooldown times. Benchmark CPU-first LLM inference and test potential GPU offloading.

*(Note: Future features such as multi-person support, dirty-lens detection, background clutter analysis, advanced pose guidance, and style personalization are explicitly excluded from this MVP.)*
