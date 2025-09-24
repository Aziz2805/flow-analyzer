# 🚀 FlowAnalyzer 

Welcome to **FlowAnalyzer**!

This app helps you extract real-time actionable flow insights from video surveillance, enabling better management of public spaces.
You can either upload a video or activate your own webcam, and the system would start running a Computer Vision **detection model** (YOLO) & a **tracker** (Botsort), showing real-time people counts evolution and the app's latency performance (FPS).

You can either choose to run a detection-only model, or a pose model. The pose model will not only detect the person but also her **keypoints**. Eich person is defined by 17 keypoints (Nose, left/right eye, shoulder, ankle, wrist etc.)

The model's size can be configurated in 5 options (n,s,m,l,x). The larger the model, the better is the accuracy, but the lower is inference speed. You should adapt this parameter based on your computing resources. 

The detection zone can be restricted by selecting 4 coordinates (in 2D format) which defines the observation zone (ROIs). You can select as many zones as you want.

Finally, detections can be csv-exported in 2 formats:

- A specific report containing frame by frame counts. Eich row represents a specific frame and the associated counts.
  
- A detailed report containing all the detections (frame by frame), where eich detected person is defined by the frame where it's been detected, her ID, her bounding-box coordinated and her keypoints coordinated (if the "Pose" task option is choosen, otherwise keypoints columns will remain blank).


**Requirements**:
- Python
- Git
- Pip

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
