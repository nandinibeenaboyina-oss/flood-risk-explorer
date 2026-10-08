# Floodline Research System

A research application for inspecting a flood-inventory CSV, preprocessing its actual columns, training classifiers, reviewing calculated test-set results, running supported scenario predictions, and viewing paper-reported figures in a separate section.

## Dataset

The specified file `datasets/India_Floods_Weka_Reduced(1).csv` was not present in the checked-out repository when this application was built. Upload the CSV in the Dataset page. The API discovers its schema, types, missing values, duplicate rows, target candidate, actual target classes, geographic columns, and parseable time fields at upload. No sample or generated event records are used by the application.

The uploaded source file is not modified. Preprocessing works on an in-memory copy. A backend restart clears the uploaded dataset and in-memory trained models; prediction history is stored in SQLite under `data/`.

## Architecture

- `frontend/`: React 19, Vite, Recharts, Leaflet, and Lucide icons.
- `backend/main.py`: FastAPI upload, schema, preview, preprocessing, ML, prediction, map, and research-data APIs.
- Classifiers: scikit-learn Decision Tree and Random Forest, plus XGBoost. Numeric inputs are median-imputed; categorical inputs are most-frequent imputed and one-hot encoded inside a training pipeline. Metrics, confusion matrices, classification reports, feature importance, and predictions come from the held-out split.
- LSTM: PyTorch univariate rainfall/flow sequence model. Training is refused unless the actual upload contains a usable time field, numeric rainfall/flow field, at least 30 valid observations, and at least 20 unique timestamps. A rainfall forecast is not presented as a severity forecast.
- Prediction history: SQLite. Dataset and trained classifier state are process-memory only.

## Paper results and live results

Fixed values transcribed from the paper are labeled **Reported in Research Paper**. Model metrics from API training are labeled **Calculated from Uploaded Dataset**. These result sources are not combined. The paper's latency table and prose conflict on scale/units; the app preserves table values and discloses that discrepancy rather than converting them.

Future-year prediction is a historical-model scenario classification: the year is metadata unless a selected model feature explicitly uses it. It is not an extrapolation of future annual risk. When the dataset has insufficient chronology, genuine LSTM forecasting is disabled with a reason. Source classes such as `S1` are not mapped to Low/Medium/High/Critical without an explicit compatible label; unknown labels do not trigger a warning level.

## Local setup

Prerequisites: Python 3.11+ (Python 3.14 is supported by the pinned dependency ranges), Node.js 20+, and npm.

### Backend

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`.

### Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite forwards `/api` requests to the backend on port 8000.

### Tests

```powershell
python -m pip install pytest httpx
python -m pytest
```

Tests use a small synthetic fixture only to verify API mechanics. Those fixture values are not shipped into the interface or used as research results.

## Production Docker deployment

The root `Dockerfile` builds the React bundle, installs the Python dependencies, and serves the single-origin application with Uvicorn on port 8000. Configure the host with Dockerfile `./Dockerfile`, build context `.`, and expose port `8000`. Mount a persistent disk at `/app/data` if prediction history must survive container restarts.

## API endpoints

- `GET /api/health`
- `POST /api/dataset/upload`
- `GET /api/dataset/info`, `GET /api/dataset/preview`, `GET /api/dataset/download`
- `POST /api/preprocess`, `GET /api/features`, `GET /api/map`
- `POST /api/train/decision-tree`, `POST /api/train/random-forest`, `POST /api/train/xgboost`, `POST /api/train/lstm`
- `POST /api/predict`, `POST /api/future-prediction`, `GET /api/models`, `GET /api/model-comparison`
- `GET /api/research-results`, `GET /api/research-results/{section}`
- `GET /api/prediction-history`, `DELETE /api/prediction-history/{id}`, `GET /api/prediction-history.csv`

All displayed warning guidance is research/application output, not an official government warning.
