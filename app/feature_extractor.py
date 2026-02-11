import librosa
import numpy as np

SR = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

def audio_to_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def extract_cnn_features(audio, cnn_model):
    mel = audio_to_mel(audio)
    mel = mel[np.newaxis, ..., np.newaxis]  # (1, 128, T, 1)

    features = cnn_model.predict(mel, verbose=0)
    return features.squeeze()  # (256,)
