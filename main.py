from pathlib import Path
from typing import Any, Dict
import logging
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Constants
# ----------------------------
MODEL_PATH = Path("model.pkl")
EXPECTED_COLS = [
    "male",
    "age",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "glucose",
    "age_sysBP",
    "BMI_glucose",
    "pulse_pressure",
    "glucose_BMI_ratio",
    "risk_burden",
]

# ----------------------------
# Load ML model
# ----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

# ----------------------------
# FastAPI instance
# ----------------------------
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts 10-year CHD risk using Logistic Regression",
    version="2.2",
)


# ----------------------------
# Input Schema (Match Training)
# ----------------------------
class UserInput(BaseModel):
    male: int = Field(..., ge=0, le=1, examples=[1])
    age: int = Field(..., gt=0, lt=120, examples=[45])
    cigsPerDay: float = Field(..., ge=0, le=80, examples=[5])
    BPMeds: int = Field(..., ge=0, le=1, examples=[0])
    prevalentStroke: int = Field(..., ge=0, le=1, examples=[0])
    prevalentHyp: int = Field(..., ge=0, le=1, examples=[1])
    diabetes: int = Field(..., ge=0, le=1, examples=[0])
    totChol: float = Field(..., gt=0, examples=[200])
    sysBP: float = Field(..., gt=0, examples=[120])
    diaBP: float = Field(..., gt=0, examples=[80])
    BMI: float = Field(..., gt=0, lt=80, examples=[25])
    glucose: float = Field(..., gt=0, examples=[90])

    @model_validator(mode="after")
    def validate_bp(self) -> "UserInput":
        if self.sysBP <= self.diaBP:
            raise ValueError("sysBP must be greater than diaBP")
        return self


# ----------------------------
# Preprocessing Function (Match Training)
# ----------------------------
def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Derived features (MUST match notebook features)
    df["age_sysBP"] = df["age"] * df["sysBP"]
    df["BMI_glucose"] = df["BMI"] * df["glucose"]
    df["pulse_pressure"] = df["sysBP"] / df["diaBP"]
    df["glucose_BMI_ratio"] = df["glucose"] / df["BMI"]
    df["risk_burden"] = df["prevalentHyp"] + df["diabetes"] + df["prevalentStroke"]

    return df.reindex(columns=EXPECTED_COLS, fill_value=0)


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "model": MODEL_PATH.name}


# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict")
def predict_heart_disease(data: UserInput):
    try:
        input_dict = data.model_dump()
        input_df = preprocess_input(input_dict)

        prob = float(model.predict_proba(input_df)[0, 1])
        pred = int(prob >= 0.5)

        content = jsonable_encoder(
            {
                "predicted_category": pred,
                "probability": round(prob, 4),
                "input_received": input_dict,
            }
        )
        return JSONResponse(status_code=200, content=content)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=jsonable_encoder(
                {
                    "error": str(exc),
                    "input_received": data.model_dump(),
                }
            ),
        ) from exc
