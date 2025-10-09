from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import Dict, Any
import pickle
import pandas as pd
import logging

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Load ML model
# ----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------
# FastAPI instance
# ----------------------------
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts 10-year CHD risk using Logistic Regression",
    version="2.1"
)

# ----------------------------
# Input Schema (Match Training)
# ----------------------------
class UserInput(BaseModel):
    male: int = Field(..., ge=0, le=1, example=1)
    age: int = Field(..., gt=0, lt=120, example=45)
    cigsPerDay: float = Field(..., ge=0, le=80, example=5)
    BPMeds: int = Field(..., ge=0, le=1, example=0)
    prevalentStroke: int = Field(..., ge=0, le=1, example=0)
    prevalentHyp: int = Field(..., ge=0, le=1, example=1)
    diabetes: int = Field(..., ge=0, le=1, example=0)
    totChol: float = Field(..., gt=0, example=200)
    sysBP: float = Field(..., gt=0, example=120)
    diaBP: float = Field(..., gt=0, example=80)
    BMI: float = Field(..., gt=0, lt=80, example=25)
    glucose: float = Field(..., gt=0, example=90)

# ----------------------------
# Preprocessing Function (Match Training)
# ----------------------------
def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Derived features (MUST match your notebook)
    df["age_sysBP"] = df["age"] * df["sysBP"]
    df["BMI_glucose"] = df["BMI"] * df["glucose"]
    df["pulse_pressure"] = df["sysBP"] / df["diaBP"]
    df["glucose_BMI_ratio"] = df["glucose"] / df["BMI"]
    df["risk_burden"] = df["prevalentHyp"] + df["diabetes"] + df["prevalentStroke"]

    # Final column order (from training)
    expected_cols = [
        "male", "age", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp",
        "diabetes", "totChol", "sysBP", "diaBP", "BMI", "glucose",
        "age_sysBP", "BMI_glucose", "pulse_pressure", "glucose_BMI_ratio", "risk_burden"
    ]

    df = df.reindex(columns=expected_cols, fill_value=0)
    return df

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
def predict_heart_disease(data: UserInput):
    try:
        # Step 1: Convert input
        input_dict = data.model_dump()
        input_df = preprocess_input(input_dict)

        # Step 2: Predict
        prob = model.predict_proba(input_df)[0, 1]
        pred = int(prob >= 0.5)

        # Step 3: Response
        content = jsonable_encoder({
            "predicted_category": pred,
            "probability": round(prob, 4),
            "input_received": input_dict
        })

        return JSONResponse(status_code=200, content=content)

    except Exception as e:
        logging.exception("Prediction failed")
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder({
                "error": str(e),
                "input_received": data.model_dump()
            })
        )
