import cv2
import time
from perception.camera import Camera
from perception.tracker import SubjectTracker
from perception.visual import calculate_blur, calculate_brightness, detect_backlight, calculate_motion
from perception.state_builder import StateBuilder
from ui.overlay import DebugUI
from core.config import CAMERA_INDEX

def main():
    print("Initializing AI Photography Assistant (Stage 1)...")
    
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
    ui = DebugUI()
    
    # Timing and frame skipping for 15 FPS perception on a 30 FPS camera feed
    # We will just process every Nth frame
    frame_counter = 0
    PERCEPTION_INTERVAL = 2  # Assuming 30fps camera, processing every 2nd frame gives 15fps
    
    prev_frame = None
    current_state = state_builder.current_state

    print("Pipeline started. Press 'q' to quit.")
    cv2.namedWindow("Photography Agent - Debug UI", cv2.WINDOW_NORMAL)
    
    while True:
        frame = cam.read_frame()
        if frame is None:
            print("Warning: Invalid frame received. Skipping...")
            time.sleep(0.1)
            continue
            
        timestamp = time.time()
        
        # Only run heavy perception on intervals
        if frame_counter % PERCEPTION_INTERVAL == 0:
            # 1. Visual Analysis
            visual_data = {
                "blur_level": calculate_blur(frame),
                "brightness": calculate_brightness(frame),
                "motion_level": calculate_motion(prev_frame, frame) if prev_frame is not None else 0.0
            }
            
            # 2. Subject Tracking
            tracker_data = tracker.process_frame(frame)
            
            # 3. Backlight check if face is detected
            if tracker_data.get("face_detected") and tracker_data.get("face_bbox"):
                visual_data["backlight"] = detect_backlight(frame, tracker_data["face_bbox"])
                # Calculate specific face brightness
                from perception.visual import calculate_region_brightness
                visual_data["face_brightness"] = calculate_region_brightness(frame, tracker_data["face_bbox"])
            else:
                visual_data["backlight"] = False
                visual_data["face_brightness"] = None
                
            # 4. Update state
            current_state = state_builder.update(tracker_data, visual_data, timestamp)
            prev_frame = frame.copy()
            
        # Draw UI (runs every frame to keep video feed smooth)
        display_frame = ui.draw(frame, current_state)
        
        cv2.imshow("Photography Agent - Debug UI", display_frame)
        
        frame_counter += 1
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    print("Shutting down...")
    tracker.close()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
