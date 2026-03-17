import numpy as np
from app.audio_utils import preprocess_audio
from app.feature_extractor import extract_cnn_features

def predict_score(
    file_path,
    cnn_model,
    reg_model,
    scaler,
    pca
):
    audio = preprocess_audio(file_path)
    cnn_features = extract_cnn_features(audio, cnn_model)

    X = cnn_features.reshape(1, -1)

    print("Before scaling:",X.shape)

    X_scaled = scaler.transform(X)

    # Apply PCA (256 to 140)
    x_pca =pca.transform(X_scaled)
    print("After pca:",x_pca.shape)

    score = reg_model.predict(x_pca)[0][0]
    print("Raw score:", score)

    # Safety clamp
    score=score/25
    score = float(np.clip(score, 0.0, 10.0))

    return score
