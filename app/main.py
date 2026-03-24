from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid

from app.model_loader import load_models
from app.inference import predict_score

app = FastAPI(title="Audio Steganography Quality Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://audiostego.netlify.app",
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lstm_model, xgb_model, lgbm_model, scaler = load_models()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["mp3", "wav", "flac"]:
        return {"error": "Unsupported audio format"}

    temp_name = f"{uuid.uuid4()}.{ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_name)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        score, interpretation = predict_score(
            temp_path, lstm_model, xgb_model, lgbm_model, scaler)
    finally:
        os.remove(temp_path)

    return {
        "predicted_score":            round(score, 2),
        "scale":                      "0–10",
        "grade":                      interpretation["grade"],
        "steganography_suitability":  interpretation["steganography_suitability"],
        "description":                interpretation["description"],
        "detectability_risk":         interpretation["detectability_risk"],
        "perceptual_distortion":      interpretation["perceptual_distortion"],
        "interpretation":             "Higher score indicates better steganography quality"
    }
