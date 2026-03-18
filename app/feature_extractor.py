import librosa
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

SR         = 16000
N_MELS     = 128
N_FFT      = 1024
HOP_LENGTH = 256

# Load MERT once at module level
_device    = "cuda" if torch.cuda.is_available() else "cpu"
_processor = AutoProcessor.from_pretrained(
                "m-a-p/MERT-v1-95M", trust_remote_code=True)
_mert      = AutoModel.from_pretrained(
                "m-a-p/MERT-v1-95M", trust_remote_code=True).to(_device)
_mert.eval()

def extract_features(audio, lstm_model):
    # ── LSTM embeddings (128 dims) ──
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max).T  # (251, 128)
    mel_input = mel_db[np.newaxis, :, :]              # (1, 251, 128)

    from tensorflow.keras import models
    extractor = models.Model(
        inputs=lstm_model.input,
        outputs=lstm_model.get_layer("embedding").output)
    lstm_feats = extractor.predict(mel_input, verbose=0).squeeze()  # (128,)

    # ── MERT embeddings (768 dims) ──
    inputs = _processor(
        [audio], sampling_rate=SR,
        return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        out = _mert(**inputs, output_hidden_states=True)
    mert_feats = out.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()  # (768,)

    # ── Combine → 896 dims ──
    return np.concatenate([lstm_feats, mert_feats])
