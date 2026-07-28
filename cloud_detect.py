import cv2
import numpy as np
import onnxruntime as ort
import time


# ============================
# MODEL SETTINGS
# ============================

MODEL_PATH = "weights.onnx"

IMG_SIZE = 640

CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45


CLASS_NAMES = [
    "altocumulus",
    "altostratus",
    "cirrocumulus",
    "cirrostratus",
    "cirrus",
    "cumulonimbus",
    "cumulus",
    "nimbostratus",
    "stratocumulus",
    "stratus"
]


# ============================
# LOAD MODEL
# ============================

print("Loading model...")

options = ort.SessionOptions()

options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

# Let ONNX Runtime choose threading
options.intra_op_num_threads = 0
options.inter_op_num_threads = 0


session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=options,
    providers=[
        "CPUExecutionProvider"
    ]
)

input_name = session.get_inputs()[0].name

print("Model loaded")
print("Input:", session.get_inputs()[0].shape)
print("Classes:", session.get_modelmeta().custom_metadata_map.get("names"))



# ============================
# PREPROCESS
# ============================

def preprocess(frame):

    img = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(np.float32) / 255.0

    img = np.transpose(
        img,
        (2, 0, 1)
    )

    img = np.expand_dims(
        img,
        axis=0
    )

    return img



# ============================
# POSTPROCESS
# YOLO26 end2end output:
# x1,y1,x2,y2,confidence,class_id
# ============================

def postprocess(output, shape):

    predictions = np.squeeze(output[0])

    h, w = shape[:2]

    boxes = []
    scores = []
    class_ids = []


    for det in predictions:

        x1, y1, x2, y2, confidence, class_id = det


        if confidence < CONF_THRESHOLD:
            continue


        # normalized coordinates

        if x2 <= 1.0 and y2 <= 1.0:

            x1 *= w
            x2 *= w
            y1 *= h
            y2 *= h


        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)


        boxes.append(
            [
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ]
        )

        scores.append(
            float(confidence)
        )

        class_ids.append(
            int(class_id)
        )


    detections = []


    if len(boxes) == 0:
        return detections


    indexes = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )


    if len(indexes) > 0:

        for i in indexes.flatten():

            detections.append(
                (
                    boxes[i],
                    scores[i],
                    class_ids[i]
                )
            )


    return detections



# ============================
# CAMERA
# ============================

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    480
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    360
)


if not camera.isOpened():

    print("Camera not found")
    exit()



# ============================
# MAIN LOOP
# ============================

frame_count = 0

last_detections = []

previous_time = time.time()


while True:

    ret, frame = camera.read()

    if not ret:
        break


    frame_count += 1


    # Run AI every 3 frames

    if frame_count % 3 == 0:

        input_tensor = preprocess(frame)


        outputs = session.run(
            None,
            {
                input_name: input_tensor
            }
        )


        last_detections = postprocess(
            outputs,
            frame.shape
        )


    detections = last_detections



    # Draw detections

    for box, confidence, class_id in detections:

        x, y, bw, bh = box


        if class_id < len(CLASS_NAMES):
            label = CLASS_NAMES[class_id]
        else:
            label = f"class_{class_id}"


        cv2.rectangle(
            frame,
            (x, y),
            (x + bw, y + bh),
            (0,255,0),
            2
        )


        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )



    # FPS

    now = time.time()

    fps = 1 / (now - previous_time)

    previous_time = now


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )


    cv2.imshow(
        "Cloud Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



camera.release()
cv2.destroyAllWindows()
