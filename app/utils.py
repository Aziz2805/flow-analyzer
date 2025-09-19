import cv2
from ultralytics import YOLO


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

KPT_NAMES = [
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee',
    'R_Knee', 'L_Ankle', 'R_Ankle'
]

def get_frame_stats(results, frame_index):
    """Retourne une liste de dictionnaires avec les stats frame-par-frame"""
    frame_data = []

    r = results[0]
    boxes, keypoints = r.boxes, r.keypoints

    if boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()
        boxes_xyxy = boxes.xyxy.float().cpu().tolist()
        kpts_list = keypoints.xy.cpu().tolist() if keypoints is not None else [None] * len(ids)

        for tid, (x1, y1, x2, y2), kp in zip(ids, boxes_xyxy, kpts_list):
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            person_data = {
                'frame_index': frame_index,
                'track_id': tid,
                'bbox_x': x,
                'bbox_y': y,
                'bbox_w': w,
                'bbox_h': h,
                **{f"{name}_x": None for name in KPT_NAMES},
                **{f"{name}_y": None for name in KPT_NAMES},
            }

            if kp:
                for idx, (xk, yk) in enumerate(kp):
                    person_data[f"{KPT_NAMES[idx]}_x"] = None if xk == 0.0 and yk == 0.0 else xk
                    person_data[f"{KPT_NAMES[idx]}_y"] = None if xk == 0.0 and yk == 0.0 else yk

            frame_data.append(person_data)

    return frame_data