import cv2
from ultralytics import YOLO

def load_model(task: str, size: str):
    """
    Construit dynamiquement le chemin du modèle YOLO
    task : 'detect' ou 'pose'
    size : 'n', 's', 'm', 'l', 'x'
    """
    model_path = f"app/models/yolo11{size}.onnx" if task == "detect" else f"app/models/yolo11{size}-pose.onnx"
    model = YOLO(model_path)
    return model

def predict(cap, task: str = "detect", size: str = "n"):
    model = load_model(task, size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[0]) 
        annotated_frame = results[0].plot()

        _, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    cap.release()
