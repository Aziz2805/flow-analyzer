from ultralytics import YOLO, RTDETR

model = YOLO('yolo11l-custom.pt')
model.export(format = "onnx")


"""
for i in {'n', 's', 'm', 'l', 'x'}:
    
    detect_model = YOLO(f'yolo11{i}.pt')
    detect_model.export(format="onnx")

    pose_model = YOLO(f'yolo11{i}-pose.pt')
    pose_model.export(format="onnx")
"""