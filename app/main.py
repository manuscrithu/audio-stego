from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid

from app.model_loader import load_models
from app.inference import predict_score
from app.db import init_db, save_song_data

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


cnn_model, reg_model, scaler = load_models()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
def startup_event():
    init_db()

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
        record_id = save_song_data(file.filename, round(score, 2))
    finally:
        os.remove(temp_path)

    return {
        "id": record_id,
        "songName": file.filename,
        "predicted_score": round(score, 2),
        "scale": "0–10",
        "interpretation": "Higher score indicates better steganography quality"
    }
