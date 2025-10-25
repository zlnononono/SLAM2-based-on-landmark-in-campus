from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")  # load an official model
model = YOLO("/home/gzz/桌面/ultralytics-main/runs/detect/train30/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")
