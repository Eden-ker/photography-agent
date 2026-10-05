import cv2
import time
from perception.camera import Camera
from perception.tracker import SubjectTracker
from perception.visual import calculate_blur, calculate_brightness, detect_backlight, calculate_motion
from perception.state_builder import StateBuilder
from controller.guidance import GuidanceController
from agent.gemini_agent import GeminiAgent
from agent.worker import AgentWorker
from ui.overlay import DebugUI
from core.config import CAMERA_INDEX

def main():
    print("Initializing AI Photography Assistant (Snapshot MVP)...")
    
    try:
        tracker = SubjectTracker()
    except Exception as e:
        print(f"Error: Could not initialize MediaPipe. {e}")
        return

    try:
        cam = Camera(CAMERA_INDEX)
    except Exception as e:
        print(f"Error: Could not initialize camera. {e}")
        tracker.close()
        return
        
    state_builder = StateBuilder()
    
    # Initialize agents
    gemini_agent = GeminiAgent()
    agent_worker = AgentWorker(gemini_agent)
    
    controller = GuidanceController(agent_worker)
    ui = DebugUI()
    
    frame_counter = 0
    PERCEPTION_INTERVAL = 2
    
    prev_frame = None
    current_state = state_builder.current_state

    print("Pipeline started. Press 'SPACE' to Analyze. Press 'q' to quit.")
    cv2.namedWindow("Photography Agent - MVP", cv2.WINDOW_NORMAL)
    
    while True:
        frame = cam.read_frame()
        if frame is None:
            time.sleep(0.1)
            continue
            
        timestamp = time.time()
        
        # Perception
        if frame_counter % PERCEPTION_INTERVAL == 0:
            visual_data = {
                "blur_level": calculate_blur(frame),
                "brightness": calculate_brightness(frame),
                "motion_level": calculate_motion(prev_frame, frame) if prev_frame is not None else 0.0
            }
            
            tracker_data = tracker.process_frame(frame)
            
            if tracker_data.get("face_detected") and tracker_data.get("face_bbox"):
                visual_data["backlight"] = detect_backlight(frame, tracker_data["face_bbox"])
            else:
                visual_data["backlight"] = False
                
            current_state = state_builder.update(tracker_data, visual_data, timestamp)
            prev_frame = frame.copy()
            
            # Evaluate deterministic rules
            controller.evaluate(current_state)
            
        # Draw UI
        display_frame = ui.draw(
            frame, 
            current_state, 
            guidance_state=controller.state, 
            active_instruction=controller.active_instruction,
            target=controller.target
        )
        
        # Pad frame to match window aspect ratio to prevent stretching
        try:
            win_rect = cv2.getWindowImageRect("Photography Agent - MVP")
            if win_rect[2] > 0 and win_rect[3] > 0:
                win_w, win_h = win_rect[2], win_rect[3]
                f_h, f_w = display_frame.shape[:2]
                
                target_ratio = win_w / win_h
                frame_ratio = f_w / f_h
                
                if abs(target_ratio - frame_ratio) > 0.01:
                    if target_ratio > frame_ratio:
                        # Window is wider: pillarbox
                        new_w = int(f_h * target_ratio)
                        pad = new_w - f_w
                        left = pad // 2
                        right = pad - left
                        display_frame = cv2.copyMakeBorder(display_frame, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
                    else:
                        # Window is taller: letterbox
                        new_h = int(f_w / target_ratio)
                        pad = new_h - f_h
                        top = pad // 2
                        bottom = pad - top
                        display_frame = cv2.copyMakeBorder(display_frame, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        except Exception:
            pass

        cv2.imshow("Photography Agent - MVP", display_frame)
        frame_counter += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32: # SPACEBAR
            controller.trigger_analysis(frame, current_state)

    print("Shutting down...")
    agent_worker.shutdown()
    tracker.close()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
