import numpy as np
from app.audio_utils import preprocess_audio
from app.feature_extractor import extract_cnn_features

def predict_score(
    file_path,
    cnn_model,
    reg_model,
    scaler
):
    audio = preprocess_audio(file_path)
    cnn_features = extract_cnn_features(audio, cnn_model)

    X = cnn_features.reshape(1, -1)
    X_scaled = scaler.transform(X)

    score = reg_model.predict(X_scaled)[0][0]

    # Safety clamp
    score = float(np.clip(score, 0.0, 10.0))

    return score
