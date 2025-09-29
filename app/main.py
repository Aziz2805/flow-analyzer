from fastapi import FastAPI, Request, UploadFile, File, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil, cv2, asyncio
import time
import io
import csv
import numpy as np

from .utils import rect_to_polygon, load_model, get_frame_stats

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# KPIs
CURRENT_STATS = {
    "detections": 0,
    "fps": 0,
}

HISTORY_STATS = []
VIDEO_STATS = []

connected_clients = set()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "ok", "path": str(file_path)}


def video_generator(source="webcam", path=None, task="detect", size="n", rois=None):
    """Flux vidéo synchrone, met à jour CURRENT_STATS"""
    global CURRENT_STATS
    
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    elif source == "video" and path:
        cap = cv2.VideoCapture(path)
    else:
        raise ValueError("Source invalide : utiliser 'webcam' ou 'video' avec path.")

    # --- Chargement du modèle ---
    model = load_model(task, size)
    _last_time = time.time()
    frame_index = 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calcul FPS
        now = time.time()
        fps = 1 / (now - _last_time) if now > _last_time else 0
        _last_time = now
        CURRENT_STATS["fps"] = fps

        # Détection
        results = model.track(frame, classes=[0], verbose=False)
        frame_data = get_frame_stats(results, frame_index, rois)
        
        CURRENT_STATS["detections"] = len(frame_data)

        # Details des détections
        VIDEO_STATS.extend(frame_data)

        # Frame annotée
        annotated_frame = results[0].plot()
        
        for roi in rois:
            pts = np.array(roi, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [pts] , thickness=4, color=(0,0,255), isClosed=True)

        _, buffer = cv2.imencode(".jpg", annotated_frame)

        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        
        frame_index += 1

    cap.release()


from fastapi import Query
import json

@app.get("/stream")
async def stream_video(
    source: str = "webcam",
    path: str = None,
    task: str = "detect",
    size: str = "n",
    rois: str = Query(None)
):
    rois_list = json.loads(rois) if rois else []
    
    rois_list = [[[px*3, py*3] for px, py in roi] for roi in rois_list]
    print(rois_list)
    return StreamingResponse(
        video_generator(source=source, path=path, task=task, size=size, rois=rois_list),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/detections")
async def websocket_detections(ws: WebSocket):
    """Envoie les KPIs en temps réel uniquement quand la vidéo a commencé"""
    await ws.accept()
    connected_clients.add(ws)
    try:
        while True:
            data = {
                "detections": int(CURRENT_STATS.get("detections", 0)),
                "fps": round(float(CURRENT_STATS.get("fps", 0)), 2)
            }

            # On n'envoie que si la vidéo a commencé à détecter ou fps > 0
            if data["fps"] > 0 or data["detections"] > 0:
                # Stockage historique pour le CSV
                HISTORY_STATS.append({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "counts": data["detections"],
                    "fps": data["fps"]
                })

                # Broadcast
                try:
                    await ws.send_json(data)
                except:
                    connected_clients.remove(ws)
                    break

            await asyncio.sleep(0.5)
    finally:
        connected_clients.discard(ws)


@app.get("/export")
async def export_kpis():
    """Export CSV des KPIs (counts + fps)"""
    output = io.StringIO()
    fieldnames = ["time", "counts", "fps"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in HISTORY_STATS:
        writer.writerow({
            "time": row.get("time", ""),
            "counts": int(row.get("counts", 0)),
            "fps": float(row.get("fps", 0))
        })
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=summary_report.csv"}
    )


@app.get("/export_frames")
async def export_frames():
    """Export CSV frame par frame (bbox ou pose selon la tâche)"""
    if not VIDEO_STATS:
        return {"error": "Pas de données disponibles pour l'export frame par frame"}

    output = io.StringIO()
    fieldnames = VIDEO_STATS[0].keys()  # toutes les clés du dict renvoyé par get_frame_stats
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for frame_data in VIDEO_STATS:
        writer.writerow(frame_data)
    
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=detailed_report.csv"}
    )