## Audio Steganography Quality Prediction API

This FastAPI backend predicts the overall quality of audio steganography
using CNN-extracted audio features and a trained regression model.

### Pipeline
1. MP3 upload
2. Audio resampling (16 kHz, 4 seconds)
3. Mel-spectrogram generation
4. CNN feature extraction
5. Feature normalization
6. Neural network regression
7. Score prediction (0–10)

### API Endpoint
POST /predict
