import cv2
from ultralytics import YOLO

# 1. Load the lightweight YOLOv8 nano model
# This will download the model automatically on the first run
model = YOLO('yolo26n.pt')

# Export to ONNX format
model.export(format='ncnn', imgsz=320) # Lowering resolution (e.g., to 320) boosts FPS significantly

model = YOLO('yolo26n_ncnn_model')

# 2. Initialize the webcam
# 0 is the default ID for the first connected camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting real-time detection. Press 'q' to exit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Perform inference
    # stream=True is efficient for video processing
    results = model(frame, stream=True)

    # 4. Annotate the frame with detections
    for result in results:
        annotated_frame = result.plot()

        # 5. Display the output
        cv2.imshow('YOLO Real-Time Detection', annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

