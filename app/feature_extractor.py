import librosa
import numpy as np
import torch
import functools
from transformers import AutoModel, AutoProcessor
import tensorflow as tf

SR         = 16000
N_MELS     = 128
N_FFT      = 1024
HOP_LENGTH = 256

# ── Load MERT lazily (only when first request comes in) ──
@functools.lru_cache(maxsize=1)
def _get_mert():
    device    = "cpu"  # Render has no GPU
    processor = AutoProcessor.from_pretrained(
                    "m-a-p/MERT-v1-95M",
                    trust_remote_code=True)
    model     = AutoModel.from_pretrained(
                    "m-a-p/MERT-v1-95M",
                    trust_remote_code=True).to(device)
    model.eval()
    return processor, model, device

# ── Build extractor once, cache it ──
@functools.lru_cache(maxsize=1)
def _get_extractor(lstm_model):
    return tf.keras.models.Model(
        inputs=lstm_model.input,
        outputs=lstm_model.get_layer("embedding").output)

def extract_features(audio, lstm_model):
    # ── LSTM embeddings (128 dims) ──
    mel       = librosa.feature.melspectrogram(
                    y=audio, sr=SR, n_fft=N_FFT,
                    hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db    = librosa.power_to_db(mel, ref=np.max).T   # (251, 128)
    mel_input = mel_db[np.newaxis, :, :]                  # (1, 251, 128)

    extractor  = _get_extractor(lstm_model)
    lstm_feats = extractor.predict(mel_input, verbose=0).squeeze()  # (128,)

    # ── MERT embeddings (768 dims) ──
    processor, mert_model, device = _get_mert()
    inputs = processor(
                [audio], sampling_rate=SR,
                return_tensors="pt",
                padding=True).to(device)
    with torch.no_grad():
        out = mert_model(**inputs, output_hidden_states=True)
    mert_feats = out.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()  # (768,)

    # ── Combine → 896 dims ──
    return np.concatenate([lstm_feats, mert_feats])
