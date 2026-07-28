import cv2
import time
from threading import Thread
from ultralytics import YOLO

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

# 1. Load the exported NCNN folder directly
model = YOLO('best_ncnn_model', task='classify')

# 2. Start the threaded camera stream
vs = VideoStream(src=0).start()
time.sleep(1.0)  # Allow camera sensor to warm up

print("Starting optimized real-time detection. Press 'q' to exit.")

while True:
    frame = vs.read()
    if frame is None:
        continue

    # 3. Optimized Inference (Removed stream=True for single-frame classification reliability)
    results = model(frame, imgsz=320, verbose=False)
    result = results[0]

    # 4. Process Classification Results
    if result.probs is not None:
        top_class_idx = result.probs.top1
        confidence = result.probs.top1conf.item()
        class_name = result.names[top_class_idx]

        # Write text directly onto the video frame
        display_text = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Trigger logic
        if class_name == "Cumulus" and confidence > 0.80:
            print(f"🚀 Thermal cloud detected ({confidence:.2f})! Ready for EKF handoff...")

    # 5. Display the output frame
    cv2.imshow('Drone Updraft Tracker (YOLOv26n-cls)', frame)

    # 6. Smooth Break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup
vs.stop()
cv2.destroyAllWindows()
