import joblib
import keras

LSTM_MODEL_PATH = "models/lstm_feature_extractor_v4b.keras"
XGB_PATH        = "models/xgb_v4b.pkl"
LGBM_PATH       = "models/lgbm_v4b.pkl"
SCALER_PATH     = "models/scaler_v4b.pkl"

def load_models():
    lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
    xgb_model  = joblib.load(XGB_PATH)
    lgbm_model = joblib.load(LGBM_PATH)
    scaler     = joblib.load(SCALER_PATH)
    return lstm_model, xgb_model, lgbm_model, scaler