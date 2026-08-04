from ultralytics import YOLO

# 1. Load your PyTorch model
model = YOLO("/home/raspberry/rasp-pi-code/soar-umd-drone-code/yolo26n.pt")

# 2. Export to INT8 quantized TFLite format for OpenMV MicroPython
model.export(format="tflite", int8=True, imgsz=320)
