from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid

from app.model_loader import load_models
from app.inference import predict_score

app = FastAPI(title="Audio Steganography Quality Predictor")

cnn_model, reg_model, scaler = load_models()

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
        score = predict_score(
            temp_path,
            cnn_model,
            reg_model,
            scaler
        )
    finally:
        os.remove(temp_path)

    return {
        "predicted_score": round(score, 2),
        "scale": "0–10",
        "interpretation": "Higher score indicates better steganography quality"
    }
