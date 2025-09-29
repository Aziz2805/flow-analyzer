from ultralytics import YOLO
import os

export_path = os.path.join("app", "models", "weights")
os.makedirs(export_path, exist_ok=True)

variants = ['n', 's', 'm', 'l', 'x']

for v in variants:
    for model_type in ["", "-pose"]:
        model_name = f"yolov8{v}{model_type}.pt"
        model = YOLO(model_name)
        model.export(format="onnx")
        os.remove(model_name)
        os.replace(f"yolov8{v}{model_type}.onnx", os.path.join(export_path, f"yolov8{v}{model_type}.onnx"))