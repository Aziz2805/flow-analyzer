# 🚀 FlowAnalyzer

Welcome to **FlowAnalyzer** – your tool to extract real-time, actionable insights from video surveillance and manage public spaces smarter! 👀  

## 🎥 How it works
Upload a video or use your webcam, and **FlowAnalyzer** runs:  
- A **detection model** (YOLO)  
- A **tracker** (Botsort)  

See **real-time people counts** and app performance (FPS) as it happens. ⚡  

## 🧍 Models
- **Detection-only:** Detects people in the scene.  
- **Pose model:** Detects people **and their 17 keypoints** (nose, eyes, shoulders, wrists, ankles, etc.) for advanced analysis.  

Model sizes: **n, s, m, l, x** – larger models = higher accuracy, slower speed. Choose based on your hardware. 💻  

## 📍 Detection Zones
Restrict detection to specific areas using **4 2D coordinates** (ROIs).  
You can define **as many zones as you want**. ✅  

## 📊 Export Results
Export detection results in **CSV** format:  
- **Summary report:** frame-by-frame people counts.  
- **Detailed report:** all detections per frame, including **person ID, bounding box, and keypoints** (if Pose model selected).  


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
