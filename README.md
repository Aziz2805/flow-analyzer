# 🚀 FlowAnalyzer

Welcome to FlowAnalyzer!

FlowAnalyzer helps you extract real-time, actionable flow insights from video surveillance, enabling smarter management of public spaces. You can upload a video or use your webcam, and the system will run a detection model (YOLO) and a tracker (Botsort), displaying real-time people counts and latency performance (FPS).

You can choose between a detection-only model or a pose model. The pose model detects each person’s keypoints (17 in total: nose, eyes, shoulders, wrists, ankles, etc.) in addition to their presence.

The model comes in five sizes (n, s, m, l, x). Larger models offer higher accuracy but slower inference speed. Choose the size according to your computing resources.

Detection zones can be restricted by specifying four 2D coordinates to define Regions of Interest (ROIs). You can select as many zones as you need.

Finally, detection results can be exported as CSV in two formats:

- Summary report: frame-by-frame people counts, one row per frame.
- Detailed report: all detections per frame, including person ID, bounding box coordinates, and keypoints (if the Pose model is selected; otherwise keypoint columns remain empty).


**Installation instructions:**

- Clone the respository & navigate to it:
```
git clone https://github.com/Aziz2805/flow-analyzer
cd flow-analyzer
```

The project structure should look like this:

```
flow-analyzer/
│── app/
│   ├── main.py          # FastAPI entrypoint
│   ├── models/          # Model exports & ONNX files
│   └── utils/           # Helper functions
│── requirements.txt
│── README.md
```

- Create a virtual environment & activate it
```
python -m venv flow-analyzer-env
flow-analyzer-env\Scripts\activate 
```
- Upgrade pip & install requirements
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```
- 🧠 Load vision models in ONNX format
```
python app/models/export_onnx.py
```
- ▶️ Start the app
```
uvicorn app.main:app --reload
```

Then open your browser at 👉 http://127.0.0.1:8000

👉 Here you go! 🎉
