import librosa
import numpy as np

SR         = 16000
DURATION   = 4
TARGET_LEN = SR * DURATION  # 64000 samples

def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SR, mono=True)

    if len(audio) < TARGET_LEN:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    else:
        audio = audio[:TARGET_LEN]

    return audio.astype(np.float32)