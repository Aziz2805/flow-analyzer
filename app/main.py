from fastapi import FastAPI, Request, UploadFile, File, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil, cv2, asyncio

from .utils import load_model, get_stats, annotate_frame
from ultralytics import solutions
import supervision as sv

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CURRENT_DETECTIONS = 0
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


def video_generator(source="webcam", path=None, task="detect", size="n"):
    """
    Génère le flux vidéo et met à jour CURRENT_DETECTIONS
    """
    global CURRENT_DETECTIONS

    # --- Choix de la source vidéo ---
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    elif source == "video" and path:
        cap = cv2.VideoCapture(path)
    else:
        raise ValueError("Source invalide : utiliser 'webcam' ou 'video' avec path.")

    # --- Chargement du modèle ---
    model = load_model(task, size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Inférence YOLO ---
        results = model.track(frame, classes=[0], persist=True)
        CURRENT_DETECTIONS = get_stats(results)

        # --- Annotation en fonction de la tâche ---
        annotated_frame = annotate_frame(frame, results, source, task)

        # --- Encodage JPEG ---
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()



@app.get("/stream")
async def stream_video(
    source: str = "webcam",
    path: str = None,
    task: str = "detect",
    size: str = "n"
):
    return StreamingResponse(
        video_generator(source, path, task, size),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/detections")
async def websocket_detections(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(1)
    except:
        pass
    finally:
        connected_clients.remove(ws)


@app.on_event("startup")
async def start_kpi_sender():
    async def send_kpis():
        while True:
            data = {"detections": CURRENT_DETECTIONS}
            for ws in connected_clients:
                try:
                    await ws.send_json(data)
                except:
                    pass
            await asyncio.sleep(0.5)

    asyncio.create_task(send_kpis())
