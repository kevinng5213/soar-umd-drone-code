import cv2
import time
import signal
import sys
from datetime import datetime
from threading import Thread
from ultralytics import YOLO

# Flag to signal thread/loops to stop cleanly when Pi powers off
running = True

def shutdown_handler(signum, frame):
    """Gracefully catches systemd/OS stop signals (SIGTERM/SIGINT) on power down."""
    global running
    print("\n[INFO] Shutdown signal received. Closing video file safely...")
    running = False

# Register signal handlers for clean systemd shutdowns
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

# --- MULTI-THREADED CAMERA CLASS ---
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        global running
        while not self.stopped and running:
            if not self.stream.isOpened():
                self.stopped = True
                break
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- MAIN INFERENCE & RECORDING PIPELINE ---

# 1. Load the exported NCNN folder directly
model = YOLO('/home/pi/best_ncnn_model', task='classify')

# 2. Start the threaded camera stream
vs = VideoStream(src=0).start()
time.sleep(1.0)  # Allow camera sensor to warm up

# Fetch frame dimensions for VideoWriter
init_frame = vs.read()
if init_frame is not None:
    frame_height, frame_width = init_frame.shape[:2]
else:
    frame_width, frame_height = 1920, 1080  # Default fallback

# 3. Initialize VideoWriter with timestamped filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"/home/pi/drone_flight_{timestamp}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, 30.0, (frame_width, frame_height))

print(f"[INFO] Started inference & saving video to: {output_filename}")

try:
    while running:
        frame = vs.read()
        if frame is None:
            continue

        # 4. Optimized Inference
        results = model(frame, imgsz=320, stream=True, verbose=False)

        for result in results:
            if result.probs is not None:
                top_class_idx = result.probs.top1
                confidence = result.probs.top1conf.item()
                class_name = result.names[top_class_idx]

                # Draw classification overlay on the frame
                display_text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if class_name == "Cumulus" and confidence > 0.80:
                    print(f"🚀 Thermal cloud detected ({confidence:.2f})!")

        # 5. Write annotated frame to the MP4 file on disk
        out.write(frame)

        # 6. Display to CONNEX HDMI transmitter output
        cv2.imshow('Drone Updraft Tracker (YOLO-cls)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("[INFO] Cleaning up hardware resources...")
    vs.stop()
    out.release()
    cv2.destroyAllWindows()
    print("[SUCCESS] Video saved cleanly to storage.")
