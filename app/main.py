from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2, shutil
from pathlib import Path
from .model import predict

app = FastAPI()

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "ok", "path": str(file_path)}

@app.get("/stream")
async def stream_video(
    source: str = "webcam",
    path: str = None,
    task: str = "detect",
    size: str = "n"
):
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    elif source == "video" and path:
        cap = cv2.VideoCapture(path)
    else:
        return JSONResponse({"error": "Invalid source"}, status_code=400)

    return StreamingResponse(
        predict(cap, task=task, size=size),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
