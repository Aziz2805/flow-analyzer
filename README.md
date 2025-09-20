# 🚀 FlowAnalyzer 

Welcome to **FlowAnalyzer**!
If you want to analyze the flow in public places, you're in the right place.

**Requirements**:
- Python
- Git
- Pip

- Clone the respository & navigate to it:
```
git clone https://github.com/Aziz2805/flow-analyzer
cd flow-analyzer
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


**Project Structure**
```
flow-analyzer/
│── app/
│   ├── main.py          # FastAPI entrypoint
│   ├── models/          # Model exports & ONNX files
│   └── utils/           # Helper functions
│── requirements.txt
│── README.md
```
👉 Here you go! 🎉
