import numpy as np
from app.audio_utils import preprocess_audio
from app.feature_extractor import extract_features

def predict_score(file_path, lstm_model, xgb_model, lgbm_model, scaler):
    audio    = preprocess_audio(file_path)
    features = extract_features(audio, lstm_model)  # (896,)

    X_scaled    = scaler.transform(features.reshape(1, -1))
    pred_xgb    = xgb_model.predict(X_scaled)[0]
    pred_lgbm   = lgbm_model.predict(X_scaled)[0]
    pred_blend  = 0.5 * pred_xgb + 0.5 * pred_lgbm

    return float(np.clip(pred_blend, 0.0, 10.0))
