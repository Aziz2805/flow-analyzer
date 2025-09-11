from ultralytics import YOLO

for i in {'n', 's', 'm', 'l', 'x'}:
    model1 = YOLO(f'yolo11{i}-pose.pt')
    model2 = YOLO(f'yolo11{i}.pt')

    model1.export(format = "onnx")
    model2.export(format = "onnx")