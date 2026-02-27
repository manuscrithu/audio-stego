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

### PostgreSQL Storage

Prediction results are stored in PostgreSQL in table `"song-data"` with:
- `id` (auto-generated, `BIGSERIAL` primary key)
- `songName` (`TEXT`)
- `score` (`DOUBLE PRECISION`)

Create a `.env` file (you can copy from `.env.example`) and set either:
- `DATABASE_URL` (recommended):
	- `postgresql://postgres:<POSTGRES_PASSWORD>@centerbeam.proxy.rlwy.net:13454/railway`

Or set individual variables:
- `DB_HOST` (for Railway: `centerbeam.proxy.rlwy.net`)
- `DB_PORT` (for Railway: `13454`)
- `DB_NAME` (for Railway: `railway`)
- `DB_USER` (for Railway: `postgres`)
- `DB_PASSWORD`
- `DB_SSLMODE` (recommended: `require`)

Table creation is handled automatically on API startup.
