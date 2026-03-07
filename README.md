# End-to-end Heart Disease Prediction Using Logistic Regression

A lightweight end-to-end project that predicts **10-year coronary heart disease (CHD) risk** using a trained Logistic Regression model.

## What is included
- **FastAPI backend** (`main.py`) for prediction serving.
- **Streamlit frontend** (`frontend.py`) for interactive inputs.
- **Trained model artifact** (`model.pkl`).
- **Training data** (`HeartDisease.csv`) and experimentation notebook.

## Quick start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3) Run the frontend
```bash
streamlit run frontend.py
```

## API usage

### Health check
```bash
curl http://127.0.0.1:8000/health
```

### Prediction example
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "male": 1,
    "age": 45,
    "cigsPerDay": 5,
    "BPMeds": 0,
    "prevalentStroke": 0,
    "prevalentHyp": 1,
    "diabetes": 0,
    "totChol": 200,
    "sysBP": 120,
    "diaBP": 80,
    "BMI": 25,
    "glucose": 90
  }'
```

## Improvement ideas
- Add a training pipeline script that saves preprocessing + model together (single reproducible artifact).
- Add automated tests for `/predict` and edge-case validation.
- Version the model and expose metadata (training date, metric snapshot) from a `/model-info` endpoint.
