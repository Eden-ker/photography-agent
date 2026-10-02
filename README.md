# AI Photography Assistant

An AI-powered local, real-time photography assistant. This application uses a webcam and computer vision to analyze a scene and subject in real-time, and will eventually provide smart framing and composition feedback using a local LLM.

## Current Architecture
The system is built on an asynchronous, decoupled architecture:
1. **Perception Loop**: (Implemented) Runs OpenCV and MediaPipe to continuously extract tracking points and build a `SceneState`.
2. **Guidance Controller**: (To be implemented) Evaluates `SceneState` to manage task lifecycles.
3. **Photography Agent**: (To be implemented) Uses a local LLM (Ollama) to reason about the scene and provide instructions.
4. **UI**: (Minimal implementation) Displays the camera feed, bounding boxes, and current state metrics.

## Installation

Ensure you have Python 3.10+ installed. To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Running Stage 1

Currently, only **Stage 1: Foundation & Perception** is implemented. 
To start the webcam perception pipeline and see the debug UI, run:

```bash
python main.py
```
*Press 'q' to quit the application.*

## What is Currently Implemented (Stage 1)
- Webcam capture using OpenCV (aiming for ~30 FPS UI).
- Real-time `SceneState` aggregation using MediaPipe Pose and Face tracking.
- Deterministic computer vision calculations for blur, brightness, and motion.
- Exponential Moving Average (EMA) smoothing for continuous perception values.
- A basic debug UI showing tracking markers and raw metrics.
- Unit tests for the deterministic CV algorithms.

## What is Intentionally Not Implemented Yet
- The local LLM integration (Ollama / Qwen2.5-1.5B).
- The Photography Agent and JSON reasoning logic.
- The Guidance Controller and state machine (task persistence, cooldown).
- Future features like multi-person tracking, dirty lens detection, etc.
