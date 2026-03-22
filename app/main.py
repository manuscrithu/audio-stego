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


cnn_model, reg_model, scaler,pca= load_models()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_interpretation(score: float) -> str:
    if score >= 8.5:
        return "Excellent carrier - Highly recommended for steganography"
    if score >= 8.0:
        return "Very good carrier - Suitable for steganography"
    if score >= 7.0:
        return "Good carrier - Acceptable for most of the use cases; but not ideal for high-security applications"
    if score >= 6.0:
        return "Moderate carrier - Use with caution; may not be ideal for sensitive data"
    if score >= 5.0:
        return "Poor carrier - Not suitable for steganography"
    return "Unsuitable - Avoid using this audio for steganography"

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
            scaler,
            pca
        )
    finally:
        os.remove(temp_path)

    return {
        "predicted_score": round(score, 2),
        "scale": "0–10",
        "interpretation": get_interpretation(score)
    }
