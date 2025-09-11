from ultralytics import YOLO

for i in {'n', 's', 'm', 'l', 'x'}:
    
    detect_model = YOLO(f'yolo11{i}.pt')
    detect_model.export(format="onnx")

    pose_model = YOLO(f'yolo11{i}-pose.pt')
    pose_model.export(format="onnx")