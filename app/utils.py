import cv2
import os
from ultralytics import YOLO

import numpy as np
from ultralytics import solutions


def load_model(task: str, size: str):

    weights_path = os.path.join("app","models","weights")
    model_name = f"yolo11{size}"

    if size != "custom":
        if task == "detect":
            model_path = os.path.join(weights_path, f"{model_name}.onnx")
        else:
            model_path = os.path.join(weights_path, f"{model_name}-pose.onnx")
    else:
        model_path = os.path.join(weights_path, f"{model_name}-custom.onnx")
    model = YOLO(model_path)

    return model


KPT_NAMES = [
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee',
    'R_Knee', 'L_Ankle', 'R_Ankle'
]

def get_zone(point, zones):
    for i, zone in enumerate(zones):
        if cv2.pointPolygonTest(np.array(zone), point, False) >= 0:
            return i + 1
    return None

def rect_to_polygon(rect):
    x, y, w, h = rect
    return [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]

def get_frame_stats(results, frame_index, rois=[]):

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
            bottom_center = (x + w / 2, y + h)

            zone = get_zone(bottom_center, rois)

            if zone is not None:
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
