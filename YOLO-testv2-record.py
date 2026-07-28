import os
# Force the display to the Pi's physical HDMI port for the CONNEX transmitter
os.environ["DISPLAY"] = ":0"

import cv2
import time
import threading
from datetime import datetime
from ultralytics import YOLO

# Global variable to capture SSH terminal input
exit_command = None

def terminal_listener():
    """Listens for terminal commands in the background so SSH keystrokes work."""
    global exit_command
    print("\n--- CONTROL MENU ---")
    print("Type 'q' + Enter to QUIT normally.")
    print("Type 'r' + Enter to QUIT and SAVE the flight video.")
    print("--------------------\n")
    while True:
        cmd = input().strip().lower()
        if cmd in ['q', 'r']:
            exit_command = cmd
            break

# --- MULTI-THREADED CAMERA CLASS ---
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.stream.isOpened():
                self.stopped = True
                break
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- MAIN INFERENCE PIPELINE ---

model = YOLO('best_ncnn_model', task='classify')

# Start the threaded camera stream
vs = VideoStream(src=0).start()
time.sleep(1.0)  

# Start the SSH terminal keyboard listener
threading.Thread(target=terminal_listener, daemon=True).start()

# --- VIDEO RECORDING SETUP ---
# Read one frame to get the camera's resolution
test_frame = vs.read()
height, width = test_frame.shape[:2]

# Setup the VideoWriter to save a temporary background file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
temp_video_path = "temp_flight_recording.mp4"
# Using 15 FPS as a safe estimate for Pi 5 inference speeds; adjust if playback is too fast/slow
video_writer = cv2.VideoWriter(temp_video_path, fourcc, 15.0, (width, height))


while True:
    frame = vs.read()
    if frame is None:
        continue

    # Optimized Inference
    results = model(frame, imgsz=320, verbose=False)
    result = results[0]

    # Process Classification Results
    if result.probs is not None:
        top_class_idx = result.probs.top1
        confidence = result.probs.top1conf.item()
        class_name = result.names[top_class_idx]

        display_text = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if class_name == "Cumulus" and confidence > 0.80:
            print(f"🚀 Thermal cloud detected ({confidence:.2f})! Ready for EKF handoff...")

    # Write the frame to the background video file
    video_writer.write(frame)

    # Display the output frame to the CONNEX transmitter
    cv2.imshow('Drone Updraft Tracker (YOLOv26n-cls)', frame)

    # We still need waitKey(1) to let the OpenCV GUI render the image properly
    cv2.waitKey(1) 

    # Check if the user typed 'q' or 'r' in the SSH terminal
    if exit_command:
        break

# --- CLEANUP AND SAVING ---
print("\nShutting down camera and closing windows...")
vs.stop()
video_writer.release()
cv2.destroyAllWindows()

# Handle the video file based on how you exited
if exit_command == 'r':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_video_path = f"flight_log_{timestamp}.mp4"
    os.rename(temp_video_path, final_video_path)
    print(f"✅ Video successfully saved as: {final_video_path}")
else:
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    print("❌ Exited normally. Video discarded.")
