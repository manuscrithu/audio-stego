# import numpy as np
# from app.audio_utils import preprocess_audio
# from app.feature_extractor import extract_features
#
# def predict_score(file_path, lstm_model, xgb_model, lgbm_model, scaler):
#     audio    = preprocess_audio(file_path)
#     features = extract_features(audio, lstm_model)  # (896,)
#
#     X_scaled    = scaler.transform(features.reshape(1, -1))
#     pred_xgb    = xgb_model.predict(X_scaled)[0]
#     pred_lgbm   = lgbm_model.predict(X_scaled)[0]
#     pred_blend  = 0.5 * pred_xgb + 0.5 * pred_lgbm
#
#     return float(np.clip(pred_blend, 0.0, 10.0))
import numpy as np
from app.audio_utils import preprocess_audio
from app.feature_extractor import extract_features

def get_interpretation(score):
    if score >= 8.0:
        return {
            "grade": "Excellent",
            "steganography_suitability": "Highly Suitable",
            "description": "Audio quality is excellently preserved. Steganographic embedding is virtually undetectable. Ideal for covert communication.",
            "detectability_risk": "Very Low",
            "perceptual_distortion": "Minimal"
        }
    elif score >= 7.5:
        return {
            "grade": "Very Good",
            "steganography_suitability": "Very Suitable",
            "description": "Audio quality is very well preserved. Embedding is highly imperceptible to human listeners.",
            "detectability_risk": "Low",
            "perceptual_distortion": "Very Low"
        }
    elif score >= 7.0:
        return {
            "grade": "Good",
            "steganography_suitability": "Suitable",
            "description": "Audio quality is adequately preserved. Minor imperceptible distortions may exist but embedding remains hidden.",
            "detectability_risk": "Moderate",
            "perceptual_distortion": "Low"
        }
    elif score >= 6.0:
        return {
            "grade": "Fair",
            "steganography_suitability": "Marginally Suitable",
            "description": "Noticeable quality degradation. Embedding may introduce slight audible artifacts.",
            "detectability_risk": "Moderate-High",
            "perceptual_distortion": "Moderate"
        }
    else:
        return {
            "grade": "Poor",
            "steganography_suitability": "Not Suitable",
            "description": "Significant quality degradation detected. Steganographic embedding is likely detectable.",
            "detectability_risk": "High",
            "perceptual_distortion": "Significant"
        }

def predict_score(file_path, lstm_model, xgb_model, lgbm_model, scaler):
    audio    = preprocess_audio(file_path)
    features = extract_features(audio, lstm_model)  # (896,)

    X_scaled   = scaler.transform(features.reshape(1, -1))
    pred_xgb   = xgb_model.predict(X_scaled)[0]
    pred_lgbm  = lgbm_model.predict(X_scaled)[0]
    pred_blend = 0.5 * pred_xgb + 0.5 * pred_lgbm
    score      = float(np.clip(pred_blend, 0.0, 10.0))

    return score, get_interpretation(score)