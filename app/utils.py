import cv2
from ultralytics import YOLO
import supervision as sv


def load_model(task: str, size: str):
    if size != "custom":
        if task == "detect":
            model_path = f"app/models/yolo11{size}.onnx" 
        else:
            model_path = f"app/models/yolo11{size}-pose.onnx"
    else:
        model_path = f"app/models/yolo11l-custom.onnx"

    model = YOLO(model_path)
    return model


def get_stats(results):
    CURRENT_DETECTIONS = len(results[0].boxes)
    return CURRENT_DETECTIONS

# ⚡️ Initialise une seule fois (hors fonction) pour garder l'état du tracker
tracker = sv.ByteTrack()
label_annotator = sv.LabelAnnotator()

def annotate_frame(frame, results, source, task):
    """
    Annoter un frame selon la tâche (detect ou pose) avec supervision.
    """
    if task == "detect":
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id}"
            for tracker_id in detections.tracker_id
        ]

        # Choix de l’annotateur selon la source
        annotator = sv.EllipseAnnotator() if source == "video" else sv.BoxAnnotator()
        annotated_frame = annotator.annotate(scene=frame.copy(), detections=detections)

        # Ajout des labels
        return label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

    elif task == "pose":
        
        key_points = sv.KeyPoints.from_ultralytics(results[0])
        annotator = sv.EdgeAnnotator(color=sv.Color.GREEN)
        annotated_frame = annotator.annotate(scene=frame.copy(), key_points=key_points)

        return annotated_frame

    else:
        raise ValueError(f"Tâche inconnue : {task}")