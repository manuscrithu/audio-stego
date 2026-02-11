import joblib
import tensorflow as tf

CNN_MODEL_PATH = "models/cnn_feature_extractor.keras"
REG_MODEL_PATH = "models/nn_overall_score_model.keras"
SCALER_PATH    = "models/scaler.pkl"

def load_models():
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    reg_model = tf.keras.models.load_model(REG_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return cnn_model, reg_model, scaler
