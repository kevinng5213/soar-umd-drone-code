import cv2
import time
from threading import Thread
from ultralytics import YOLO
from flask import Flask, Response
import threading

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


# --- FLASK MJPEG STREAMING SERVER ---
app = Flask(__name__)
latest_frame = None  # global shared frame


def mjpeg_generator():
    global latest_frame
    while True:
        if latest_frame is None:
            continue
        ret, jpeg = cv2.imencode('.jpg', latest_frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/video')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def start_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


# --- MAIN INFERENCE PIPELINE ---
model = YOLO('best_ncnn_model', task='classify')

vs = VideoStream(src=0).start()
time.sleep(1.0)

print("Starting optimized real-time detection. View stream at:")
print("   http://<raspberry-pi-ip>:5000/video")
print("Press Ctrl+C to exit.")

# Start Flask server in background thread
threading.Thread(target=start_flask, daemon=True).start()

while True:
    frame = vs.read()
    if frame is None:
        continue

    results = model(frame, imgsz=320, stream=True, verbose=False)

    for result in results:
        if result.probs is not None:
            top_class_idx = result.probs.top1
            confidence = result.probs.top1conf.item()
            class_name = result.names[top_class_idx]

            display_text = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, display_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if class_name == "Cumulus" and confidence > 0.80:
                print(f"🚀 Thermal cloud detected ({confidence:.2f})! Ready for EKF handoff...")

    # Update global frame for streaming
    latest_frame = frame.copy()
