import cv2
import time
from threading import Thread
from ultralytics import YOLO

# --- MULTI-THREADED CAMERA CLASS ---
# Prevents cv2.VideoCapture from blocking the main inference loop
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # Set buffer size to 1 so we always fetch the newest frame
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

# 1. Load the exported NCNN folder directly (Do NOT load the .pt or export inside this loop)
# Note: Ensure you've already run `model.export(format='ncnn', imgsz=320)` once previously.
model = YOLO('yolo26n_ncnn_model', task='detect')

# 2. Start the threaded camera stream
vs = VideoStream(src=0).start()
time.sleep(1.0)  # Allow camera sensor to warm up

print("Starting optimized real-time detection. Press 'q' to exit.")

while True:
    frame = vs.read()
    if frame is None:
        continue

    # 3. Optimized Inference
    # - persist=True tracks frames in sequence efficiently
    # - verbose=False stops heavy terminal logging which slows down python loops
    results = model(frame, imgsz=320, stream=True, verbose=False)

    for result in results:
        # 4. Optional: Skip plotting bounding boxes if running completely headless on the drone
        annotated_frame = result.plot()
        
        # 5. Display the output
        cv2.imshow('YOLO Real-Time Detection', annotated_frame)

    # 6. Smooth Break (Use 10-30ms to let the GUI main loop breath)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup
vs.stop()
cv2.destroyAllWindows()
