from __future__ import annotations

import io
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT / "frontend" / "dist"
DATABASE = ROOT / "data" / "prediction_history.sqlite3"
app = FastAPI(title="Floodline Research API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

frame: pd.DataFrame | None = None
original_filename: str | None = None
processed_frame: pd.DataFrame | None = None
preprocessing_config: dict[str, Any] = {}
trained_models: dict[str, dict[str, Any]] = {}
trained_results: dict[str, dict[str, Any]] = {}
lstm_model: dict[str, Any] | None = None


class PreprocessRequest(BaseModel):
    target: str | None = None
    features: list[str] | None = None
    drop_duplicates: bool = True
    missing_strategy: str = "impute"
    encode_categoricals: bool = False
    normalize_numeric: bool = False


class TrainRequest(BaseModel):
    target: str | None = None
    features: list[str] | None = None
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = 42
    max_depth: int | None = Field(default=8, ge=1, le=40)
    n_estimators: int = Field(default=200, ge=10, le=2000)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)


class PredictRequest(BaseModel):
    model: str
    features: dict[str, Any]
    future_year: int | None = None
    state: str | None = None
    district: str | None = None


class LstmRequest(BaseModel):
    time_column: str | None = None
    rainfall_column: str | None = None
    lookback: int = Field(default=7, ge=2, le=60)
    horizon: int = Field(default=7, ge=1, le=30)
    epochs: int = Field(default=50, ge=1, le=300)
    units: int = Field(default=64, ge=4, le=256)


PAPER_RESULTS = {
    "classification": [
        {"model": "Random Forest", "accuracy": 64.56, "precision": 0.4754, "recall": 0.3894, "f1": 0.3619},
        {"model": "XGBoost", "accuracy": 65.82, "precision": 0.5008, "recall": 0.4268, "f1": 0.4387},
        {"model": "LSTM", "accuracy": 55.13, "precision": 0.4034, "recall": 0.4179, "f1": 0.4007},
    ],
    "training_validation": [
        {"model": "Decision Tree", "training_accuracy": 93.44, "validation_accuracy": 52.16},
        {"model": "Random Forest", "training_accuracy": 100, "validation_accuracy": 64.56},
        {"model": "XGBoost", "training_accuracy": 100, "validation_accuracy": 65.82},
        {"model": "LSTM", "training_accuracy": 65.92, "validation_accuracy": 55.13},
    ],
    "case_study": [
        {"attribute": "Human Fatalities", "reported_value": "28", "prediction_value": "26"},
        {"attribute": "Human Displacement", "reported_value": "176,715", "prediction_value": "180,000"},
        {"attribute": "Crop Loss (hectares)", "reported_value": "172,308 hectares", "prediction_value": "170,500 hectares"},
        {"attribute": "Severity", "reported_value": "High", "prediction_value": "High"},
    ],
    "ablation": [
        {"configuration": "Full Feature Set", "accuracy": 64.56, "latency": 0.000816},
        {"configuration": "Without Duration (Days)", "accuracy": 63.29, "latency": 0.000816},
        {"configuration": "Without Human Displaced", "accuracy": 51.90, "latency": 0.000597},
        {"configuration": "Without Area Affected", "accuracy": 59.49, "latency": 0.000573},
        {"configuration": "Without Human Fatality", "accuracy": 55.70, "latency": 0.000629},
    ],
    "latency": [
        {"component": "Classification Model", "seconds": 1.125},
        {"component": "Forecasting Model (LSTM)", "seconds": 0.496},
        {"component": "Combined Pipeline", "seconds": 1.668},
    ],
    "usability": [
        {"metric": "Task Completion Rate (%)", "mean": 91.3, "sd": 6.2, "unit": "%"},
        {"metric": "Time on Task – Severity", "mean": 38.4, "sd": 11.7, "unit": "seconds"},
        {"metric": "Time on Task – Alerts", "mean": 62.1, "sd": 18.3, "unit": "seconds"},
        {"metric": "System Usability Scale", "mean": 78.5, "sd": 9.1, "unit": "points"},
        {"metric": "Net Promoter Score", "mean": 42, "sd": None, "unit": "NPS"},
    ],
}


def _db() -> sqlite3.Connection:
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        model TEXT NOT NULL,
        input_json TEXT NOT NULL,
        predicted_severity TEXT NOT NULL,
        probability REAL,
        risk_level TEXT NOT NULL,
        future_year INTEGER,
        state TEXT,
        district TEXT
    )""")
    connection.commit()
    return connection


def _current() -> pd.DataFrame:
    if frame is None:
        raise HTTPException(409, "No dataset loaded. Upload a CSV to continue.")
    return frame


def _norm(value: Any) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _target_guess(data: pd.DataFrame) -> str | None:
    preferred = {"severity", "floodseverity", "severityclass", "class", "target", "label"}
    return next((column for column in data.columns if _norm(column) in preferred), None)


def _find_column(data: pd.DataFrame, names: set[str]) -> str | None:
    return next((column for column in data.columns if _norm(column) in names), None)


def _temporal_candidates(data: pd.DataFrame) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for column in data.columns:
        name = _norm(column)
        if not any(token in name for token in ("date", "time", "year", "timestamp", "datetime")):
            continue
        parsed = _parse_time_column(data[column], str(column))
        valid = parsed.notna()
        unique = int(parsed[valid].nunique())
        if int(valid.sum()) >= 2:
            candidates.append({"column": str(column), "valid_values": int(valid.sum()), "unique_timestamps": unique, "parseable": True})
    return candidates


def _parse_time_column(values: pd.Series, column: str) -> pd.Series:
    if "year" in _norm(column) and pd.api.types.is_numeric_dtype(values):
        years = pd.to_numeric(values, errors="coerce")
        year_strings = years.where(years.between(1800, 2200) & years.mod(1).eq(0)).astype("Int64").astype("string")
        return pd.to_datetime(year_strings, errors="coerce", format="%Y", utc=True)
    return pd.to_datetime(values, errors="coerce", format="mixed", utc=True)


def _rainfall_candidates(data: pd.DataFrame) -> list[str]:
    keywords = ("rain", "precip", "riverlevel", "waterlevel", "discharge", "streamflow")
    return [str(column) for column in data.columns if any(keyword in _norm(column) for keyword in keywords) and pd.api.types.is_numeric_dtype(data[column])]


def _time_series_status(data: pd.DataFrame) -> dict[str, Any]:
    time_columns = _temporal_candidates(data)
    rainfall_columns = _rainfall_candidates(data)
    valid_pair = None
    for time_candidate in time_columns:
        for rainfall_column in rainfall_columns:
            observations = pd.DataFrame({
                "time": _parse_time_column(data[time_candidate["column"]], time_candidate["column"]),
                "value": pd.to_numeric(data[rainfall_column], errors="coerce"),
            }).dropna().groupby("time", as_index=False)["value"].mean().sort_values("time")
            if "year" in _norm(time_candidate["column"]):
                year_steps = observations["time"].dt.year.diff().dropna()
                cadence_ratio = float((year_steps == 1).sum() / len(year_steps)) if not year_steps.empty else 0.0
            else:
                intervals = observations["time"].diff().dropna().dt.total_seconds().to_numpy() / 86400
                median_interval = float(np.median(intervals)) if len(intervals) else 0.0
                tolerance = max(median_interval * 0.2, 1 / 1440)
                cadence_ratio = float(np.mean(np.abs(intervals - median_interval) <= tolerance)) if median_interval > 0 else 0.0
            if len(observations) >= 30 and observations["time"].nunique() >= 20 and cadence_ratio >= 0.8:
                valid_pair = {
                    "time_column": time_candidate["column"], "rainfall_column": rainfall_column,
                    "usable_observations": int(len(observations)),
                    "unique_timestamps": int(observations["time"].nunique()),
                    "regular_cadence_ratio": cadence_ratio,
                }
                break
        if valid_pair:
            break
    if valid_pair:
        reason = "A numeric rainfall/flow series has at least 30 usable observations, 20 unique timestamps, and at least 80% regular cadence."
    elif not time_columns:
        reason = "No parseable date/year/time column was found. This dataset cannot support chronological forecasting."
    elif not rainfall_columns:
        reason = "No numeric rainfall, precipitation, river-level, or flow column was found."
    else:
        reason = "No date/rainfall pair meets the minimum observation counts and regular-cadence check for LSTM training."
    return {"suitable": valid_pair is not None, "time_columns": time_columns, "rainfall_columns": rainfall_columns, "minimum_observations": 30, "minimum_unique_timestamps": 20, "minimum_regular_cadence_ratio": 0.8, "selected_series": valid_pair, "reason": reason}


def _column_summary(data: pd.DataFrame) -> list[dict[str, Any]]:
    result = []
    for column in data.columns:
        series = data[column]
        item: dict[str, Any] = {
            "name": str(column),
            "dtype": str(series.dtype),
            "kind": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if item["kind"] == "numeric":
            numbers = pd.to_numeric(series, errors="coerce").dropna()
            item["statistics"] = {
                "min": _json_value(numbers.min()), "max": _json_value(numbers.max()),
                "mean": _json_value(numbers.mean()), "median": _json_value(numbers.median()),
                "standard_deviation": _json_value(numbers.std()),
            }
        else:
            item["frequencies"] = {str(key): int(value) for key, value in series.astype("string").value_counts(dropna=False).head(50).items()}
        result.append(item)
    return result


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value.item() if isinstance(value, np.generic) else value


def _dataset_info(data: pd.DataFrame) -> dict[str, Any]:
    numerical = [str(column) for column in data.columns if pd.api.types.is_numeric_dtype(data[column])]
    categorical = [str(column) for column in data.columns if not pd.api.types.is_numeric_dtype(data[column])]
    target = _target_guess(data)
    geographic = {
        "latitude": _find_column(data, {"latitude", "lat"}),
        "longitude": _find_column(data, {"longitude", "lon", "long", "lng"}),
        "state": _find_column(data, {"state", "statename", "region"}),
        "district": _find_column(data, {"district", "districtname"}),
    }
    severity = []
    if target:
        severity = [str(value) for value in data[target].dropna().unique()[:100]]
    temporal = _temporal_candidates(data)
    severity_distribution = []
    if target:
        severity_distribution = [
            {"label": str(label), "count": int(count)}
            for label, count in data[target].value_counts(dropna=True).items()
        ]
    year_distribution = []
    if temporal:
        time_column = temporal[0]["column"]
        values = data[time_column]
        if "year" in _norm(time_column) and pd.api.types.is_numeric_dtype(values):
            years = pd.to_numeric(values, errors="coerce").dropna().astype(int).astype(str)
        else:
            years = _parse_time_column(values, time_column).dropna().dt.year.astype(str)
        year_distribution = [
            {"year": str(year), "count": int(count)}
            for year, count in years.value_counts().sort_index().items()
        ]
    location_column = geographic["state"] or geographic["district"]
    location_distribution = []
    if location_column:
        location_distribution = [
            {"location": str(location), "count": int(count)}
            for location, count in data[location_column].value_counts().head(30).items()
        ]
    return {
        "dataset_name": original_filename or "Uploaded dataset",
        "rows": int(len(data)), "columns_count": int(len(data.columns)),
        "columns": [str(column) for column in data.columns],
        "column_details": _column_summary(data),
        "numerical_columns": numerical, "categorical_columns": categorical,
        "missing_cells": int(data.isna().sum().sum()),
        "rows_with_missing": int(data.isna().any(axis=1).sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "target_candidate": target, "target_classes": severity,
        "target_distribution": severity_distribution,
        "geographic_columns": geographic, "temporal_columns": temporal,
        "year_distribution": year_distribution,
        "location_distribution": location_distribution,
        "time_series": _time_series_status(data),
        "records_are_preview_only": False,
    }


def _active_data() -> pd.DataFrame:
    return processed_frame if processed_frame is not None else _current()


def _features_for(data: pd.DataFrame, target: str, requested: list[str] | None) -> list[str]:
    if target not in data.columns:
        raise HTTPException(422, f"Target column '{target}' is not present in the dataset.")
    features = requested or [str(column) for column in data.columns if column != target]
    unknown = [column for column in features if column not in data.columns]
    if unknown:
        raise HTTPException(422, f"Unknown feature columns: {unknown}")
    features = [column for column in features if column != target]
    if not features:
        raise HTTPException(422, "Select at least one input feature besides the target.")
    return features


def _risk_label(label: Any) -> tuple[str, str]:
    normalized = str(label).strip().lower()
    mappings = {
        "low": ("Low", "Normal Monitoring"), "medium": ("Medium", "Advisory"),
        "high": ("High", "Warning"), "critical": ("Critical", "Emergency Warning"),
    }
    if normalized in mappings:
        return mappings[normalized]
    return "Unmapped source class", "No severity mapping; follow official local guidance"


def _save_prediction(model_name: str, inputs: dict[str, Any], label: str, probability: float | None, risk: str, request: PredictRequest) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    with _db() as connection:
        cursor = connection.execute(
            "INSERT INTO predictions(timestamp,model,input_json,predicted_severity,probability,risk_level,future_year,state,district) VALUES(?,?,?,?,?,?,?,?,?)",
            (timestamp, model_name, json.dumps(inputs, default=str), label, probability, risk, request.future_year, request.state, request.district),
        )
        prediction_id = cursor.lastrowid
    return {"id": prediction_id, "timestamp": timestamp, "model": model_name, "inputs": inputs, "predicted_severity": label, "probability": probability, "risk_level": risk}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "dataset_loaded": frame is not None, "dataset_name": original_filename, "trained_models": sorted(trained_models), "lstm_trained": lstm_model is not None}


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    global frame, processed_frame, original_filename, preprocessing_config, trained_models, trained_results, lstm_model
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(415, "Upload a CSV file.")
    content = await file.read()
    if not content:
        raise HTTPException(400, "The uploaded CSV is empty.")
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(413, "CSV exceeds the 100 MB upload limit.")
    try:
        loaded = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", na_values=["?"])
    except UnicodeDecodeError:
        try:
            loaded = pd.read_csv(io.BytesIO(content), encoding="latin-1", na_values=["?"])
        except Exception as error:
            raise HTTPException(400, f"Unable to parse CSV: {error}") from error
    except Exception as error:
        raise HTTPException(400, f"Unable to parse CSV: {error}") from error
    if loaded.empty or len(loaded.columns) < 1:
        raise HTTPException(400, "CSV has no data rows or columns.")
    if loaded.columns.duplicated().any():
        raise HTTPException(400, "CSV contains duplicate column names; rename duplicates before upload.")
    frame = loaded
    processed_frame = None
    original_filename = Path(file.filename).name
    preprocessing_config = {}
    trained_models = {}
    trained_results = {}
    lstm_model = None
    return _dataset_info(loaded)


@app.get("/api/dataset/info")
def dataset_info() -> dict[str, Any]:
    return _dataset_info(_active_data())


@app.get("/api/dataset/preview")
def dataset_preview(
    offset: int = Query(default=0, ge=0), limit: int = Query(default=25, ge=1, le=250),
    search: str = "", sort_by: str | None = None, descending: bool = False,
) -> dict[str, Any]:
    data = _active_data()
    result = data
    if search:
        mask = result.astype("string").apply(lambda column: column.str.contains(search, case=False, na=False, regex=False)).any(axis=1)
        result = result.loc[mask]
    if sort_by:
        if sort_by not in result.columns:
            raise HTTPException(422, f"Unknown sort column: {sort_by}")
        result = result.sort_values(sort_by, ascending=not descending, na_position="last")
    total = len(result)
    page = result.iloc[offset:offset + limit].replace({np.nan: None})
    records = [{str(key): _json_value(value) for key, value in row.items()} for row in page.to_dict(orient="records")]
    return {"columns": [str(column) for column in data.columns], "records": records, "offset": offset, "limit": limit, "filtered_rows": total, "total_rows": len(data)}


@app.get("/api/dataset/download")
def download_dataset() -> Response:
    data = _active_data()
    stream = io.StringIO()
    data.to_csv(stream, index=False)
    return Response(stream.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="flood-dataset.csv"'})


@app.post("/api/preprocess")
def preprocess(request: PreprocessRequest) -> dict[str, Any]:
    global processed_frame, preprocessing_config
    source = _current().copy()
    before = _dataset_info(source)
    if request.drop_duplicates:
        source = source.drop_duplicates().reset_index(drop=True)
    target = request.target or _target_guess(source)
    if request.missing_strategy not in {"impute", "drop", "keep"}:
        raise HTTPException(422, "missing_strategy must be impute, drop, or keep.")
    if request.missing_strategy == "drop":
        source = source.dropna().reset_index(drop=True)
    elif request.missing_strategy == "impute":
        for column in source.columns:
            if column == target:
                continue
            if pd.api.types.is_numeric_dtype(source[column]):
                median = source[column].median()
                if pd.notna(median):
                    source[column] = source[column].fillna(median)
            elif source[column].notna().any():
                mode = source[column].mode(dropna=True)
                if not mode.empty:
                    source[column] = source[column].fillna(mode.iloc[0])
    if request.encode_categoricals:
        columns = [column for column in source.columns if column != target and not pd.api.types.is_numeric_dtype(source[column])]
        source = pd.get_dummies(source, columns=columns, dummy_na=False, dtype=int)
    if request.normalize_numeric:
        for column in source.select_dtypes(include=np.number).columns:
            if column != target:
                mean, std = source[column].mean(), source[column].std()
                if pd.notna(std) and std != 0:
                    source[column] = (source[column] - mean) / std
    processed_frame = source
    preprocessing_config = request.model_dump()
    return {"original": {"rows": before["rows"], "missing_cells": before["missing_cells"], "duplicate_rows": before["duplicate_rows"]}, "processed": {"rows": len(source), "missing_cells": int(source.isna().sum().sum()), "duplicate_rows": int(source.duplicated().sum()), "columns": [str(column) for column in source.columns]}, "target": target, "features": request.features, "configuration": preprocessing_config, "note": "Applied to an in-memory copy; the uploaded file is unchanged."}


@app.get("/api/features")
def feature_information() -> dict[str, Any]:
    data = _active_data()
    target = _target_guess(data)
    concepts = {
        "duration": ("duration", "days"), "fatalities": ("fatal", "death"),
        "displacement": ("displac", "relocat"), "crop_loss": ("crop", "agricultural"),
        "area_affected": ("area", "affected"), "district": ("district",),
        "state": ("state",), "severity": ("severity", "class"),
        "rainfall": ("rain", "precip"),
    }
    mapped = {}
    for concept, tokens in concepts.items():
        mapped[concept] = [str(column) for column in data.columns if any(token in _norm(column) for token in tokens)]
    return {"target_candidate": target, "features": _column_summary(data), "concept_matches": mapped, "time_series": _time_series_status(data)}


@app.get("/api/map")
def map_data() -> dict[str, Any]:
    data = _active_data()
    info = _dataset_info(data)
    target = info["target_candidate"]
    geo = info["geographic_columns"]
    lat, lon = geo["latitude"], geo["longitude"]
    if lat and lon:
        columns = [lat, lon] + ([target] if target else [])
        points = data[columns].dropna(subset=[lat, lon]).head(10000).rename(columns={lat: "latitude", lon: "longitude", target: "label"} if target else {lat: "latitude", lon: "longitude"})
        points = points.replace({np.nan: None}).to_dict(orient="records")
        return {"mode": "coordinates", "points": [{str(k): _json_value(v) for k, v in point.items()} for point in points], "geographic_columns": geo, "label_note": "Original target labels are displayed without assuming a severity ordering."}
    group_column = geo["district"] or geo["state"]
    if group_column:
        grouped = data.groupby(group_column, dropna=False).size().sort_values(ascending=False).head(100)
        return {"mode": "administrative", "geographic_columns": geo, "groups": [{"location": str(key), "events": int(value)} for key, value in grouped.items()], "label_note": "Counts by the actual administrative column; no coordinates are inferred."}
    return {"mode": "unavailable", "geographic_columns": geo, "message": "Not available from the current dataset: no latitude/longitude, state, or district column was found."}


def _make_classifier(model_name: str, request: TrainRequest, class_count: int) -> Any:
    if model_name == "decision-tree":
        return DecisionTreeClassifier(max_depth=request.max_depth, random_state=request.random_state, class_weight="balanced")
    if model_name == "random-forest":
        return RandomForestClassifier(n_estimators=request.n_estimators, max_depth=request.max_depth, random_state=request.random_state, class_weight="balanced", n_jobs=-1)
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise HTTPException(501, "XGBoost is not installed. Install the xgboost requirement.") from error
        return XGBClassifier(
            n_estimators=request.n_estimators, learning_rate=request.learning_rate,
            max_depth=request.max_depth or 6, subsample=request.subsample,
            colsample_bytree=request.colsample_bytree, random_state=request.random_state,
            objective="multi:softprob" if class_count > 2 else "binary:logistic",
            eval_metric="mlogloss" if class_count > 2 else "logloss", n_jobs=-1,
        )
    raise HTTPException(404, f"Unsupported classifier: {model_name}")


@app.post("/api/train/{model_name}")
def train_classifier(model_name: str, request: TrainRequest) -> dict[str, Any]:
    global trained_models, trained_results
    data = _active_data().copy()
    target = request.target or _target_guess(data)
    if not target:
        raise HTTPException(422, "Select a target column. No severity/class column could be identified automatically.")
    features = _features_for(data, target, request.features)
    usable = data[features + [target]].dropna(subset=[target]).copy()
    if len(usable) < 5:
        raise HTTPException(422, "At least five labelled rows are required to train and evaluate a classifier.")
    labels = usable[target].astype(str)
    counts = labels.value_counts()
    if len(counts) < 2:
        raise HTTPException(422, "Classification requires at least two target classes.")
    if len(counts) > len(usable) - 1:
        raise HTTPException(422, "Too many unique target labels for a holdout evaluation.")
    stratify = labels if counts.min() >= 2 and int(round(len(usable) * request.test_size)) >= len(counts) else None
    try:
        x_train, x_test, y_train_raw, y_test_raw = train_test_split(
            usable[features], labels, test_size=request.test_size, random_state=request.random_state, stratify=stratify,
        )
    except ValueError as error:
        raise HTTPException(422, f"Unable to create the requested train/test split: {error}") from error
    label_encoder = LabelEncoder().fit(y_train_raw)
    y_train = label_encoder.transform(y_train_raw)
    unseen = sorted(set(y_test_raw) - set(label_encoder.classes_))
    if unseen:
        raise HTTPException(422, f"The test split contains classes absent from training: {unseen}. Upload more examples or change the split.")
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(usable[column])]
    categorical = [column for column in features if column not in numeric]
    transformers = []
    if numeric:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if preprocessing_config.get("normalize_numeric"):
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    estimator = _make_classifier(model_name, request, len(label_encoder.classes_))
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predicted_encoded = pipeline.predict(x_test).astype(int)
    prediction_seconds = time.perf_counter() - start
    y_test = label_encoder.transform(y_test_raw)
    labels_sorted = list(range(len(label_encoder.classes_)))
    matrix = confusion_matrix(y_test, predicted_encoded, labels=labels_sorted)
    report = classification_report(y_test, predicted_encoded, labels=labels_sorted, target_names=[str(item) for item in label_encoder.classes_], output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_test, predicted_encoded)
    train_accuracy = accuracy_score(y_train, pipeline.predict(x_train))
    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        importances = getattr(pipeline.named_steps["classifier"], "feature_importances_", None)
        importance_rows = sorted(
            [{"feature": str(name), "importance": float(value)} for name, value in zip(feature_names, importances)],
            key=lambda item: item["importance"], reverse=True,
        ) if importances is not None else []
    except (AttributeError, ValueError):
        importance_rows = []
    true_labels = label_encoder.inverse_transform(y_test)
    predicted_labels = label_encoder.inverse_transform(predicted_encoded)
    probabilities = pipeline.predict_proba(x_test) if hasattr(pipeline, "predict_proba") else None
    prediction_rows = []
    for index, (_, input_row) in enumerate(x_test.iterrows()):
        prediction_rows.append({
            "row_index": int(input_row.name), "actual": str(true_labels[index]),
            "predicted": str(predicted_labels[index]),
            "correct": bool(true_labels[index] == predicted_labels[index]),
            "confidence": float(np.max(probabilities[index])) if probabilities is not None else None,
        })
    model_display = {"decision-tree": "Decision Tree", "random-forest": "Random Forest", "xgboost": "XGBoost"}[model_name]
    result = {
        "source": "Calculated from Uploaded Dataset", "status": "trained",
        "model": model_display, "dataset": original_filename,
        "instances": len(usable), "attributes": len(features) + 1,
        "target": target, "features": features, "class_labels": [str(item) for item in label_encoder.classes_],
        "split": {"method": "stratified" if stratify is not None else "random", "train_rows": len(x_train), "test_rows": len(x_test), "test_size": request.test_size, "random_state": request.random_state},
        "parameters": estimator.get_params(),
        "metrics": {
            "training_accuracy": float(train_accuracy), "accuracy": float(accuracy),
            "precision_macro": float(precision_score(y_test, predicted_encoded, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, predicted_encoded, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, predicted_encoded, average="macro", zero_division=0)),
            "correctly_classified": int(np.sum(y_test == predicted_encoded)),
            "incorrectly_classified": int(np.sum(y_test != predicted_encoded)),
        },
        "confusion_matrix": {"labels": [str(item) for item in label_encoder.classes_], "values": matrix.tolist()},
        "classification_report": report,
        "feature_importance": importance_rows[:100],
        "predictions": prediction_rows,
        "timing": {"training_seconds": training_seconds, "prediction_seconds": prediction_seconds},
        "tree_text": export_text(pipeline.named_steps["classifier"], feature_names=feature_names, max_depth=5) if model_name == "decision-tree" else None,
    }
    trained_models[model_name] = {"pipeline": pipeline, "features": features, "target": target, "encoder": label_encoder, "labels": [str(item) for item in label_encoder.classes_]}
    trained_results[model_name] = result
    return result


@app.get("/api/models")
def model_status() -> dict[str, Any]:
    return {"trained": list(trained_results.values()), "available_classifiers": ["decision-tree", "random-forest", "xgboost"], "lstm": _time_series_status(_active_data()) if frame is not None else {"suitable": False, "reason": "Upload a dataset to inspect time-series suitability."}}


@app.post("/api/train/lstm")
def train_lstm(request: LstmRequest) -> dict[str, Any]:
    global lstm_model
    data = _current()
    suitability = _time_series_status(data)
    time_column = request.time_column or (suitability["time_columns"][0]["column"] if suitability["time_columns"] else None)
    rainfall_column = request.rainfall_column or (suitability["rainfall_columns"][0] if suitability["rainfall_columns"] else None)
    if not suitability["suitable"] or not time_column or not rainfall_column:
        raise HTTPException(422, f"Current dataset does not contain sufficient chronological time-series information for genuine LSTM forecasting. {suitability['reason']}")
    if time_column not in data or rainfall_column not in data:
        raise HTTPException(422, "Selected time/rainfall columns are not present in the dataset.")
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise HTTPException(501, "PyTorch is required for LSTM training. Install the torch requirement.") from error
    series = pd.DataFrame({"time": _parse_time_column(data[time_column], time_column), "value": pd.to_numeric(data[rainfall_column], errors="coerce")}).dropna().sort_values("time")
    series = series.groupby("time", as_index=False)["value"].mean()
    if len(series) < max(30, request.lookback + 10):
        raise HTTPException(422, f"Only {len(series)} usable chronological observations remain; at least {max(30, request.lookback + 10)} are required.")
    values = series["value"].to_numpy(dtype=np.float32)
    minimum, maximum = float(values.min()), float(values.max())
    scaled = (values - minimum) / (maximum - minimum) if maximum != minimum else np.zeros_like(values)
    lookback = request.lookback
    samples = np.array([scaled[index - lookback:index] for index in range(lookback, len(scaled))], dtype=np.float32)
    targets = np.array([scaled[index] for index in range(lookback, len(scaled))], dtype=np.float32)
    split = max(1, int(len(samples) * 0.8))
    x_train = torch.tensor(samples[:split, :, None])
    y_train = torch.tensor(targets[:split, None])
    x_val = torch.tensor(samples[split:, :, None])
    y_val = torch.tensor(targets[split:, None])

    class RainLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=request.units, batch_first=True)
            self.output = nn.Linear(request.units, 1)

        def forward(self, values: Any) -> Any:
            sequence, _ = self.lstm(values)
            return self.output(sequence[:, -1, :])

    torch.manual_seed(42)
    network = RainLSTM()
    optimizer = torch.optim.Adam(network.parameters())
    loss_function = nn.MSELoss()
    history = []
    start = time.perf_counter()
    for _ in range(request.epochs):
        network.train()
        optimizer.zero_grad()
        loss = loss_function(network(x_train), y_train)
        loss.backward()
        optimizer.step()
        network.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(network(x_val), y_val)) if len(x_val) else None
        history.append({"training_loss": float(loss.detach()), "validation_loss": validation_loss})
    training_seconds = time.perf_counter() - start
    network.eval()
    with torch.no_grad():
        val_predictions = network(x_val).squeeze(-1).numpy() if len(x_val) else np.array([])
    actual = values[lookback + split:]
    val_prediction_values = val_predictions * (maximum - minimum) + minimum if maximum != minimum else np.full_like(val_predictions, minimum)
    last_window = scaled[-lookback:].copy()
    forecast = []
    for step in range(request.horizon):
        with torch.no_grad():
            next_scaled = float(network(torch.tensor(last_window[None, :, None])).item())
        next_value = next_scaled * (maximum - minimum) + minimum if maximum != minimum else minimum
        forecast.append({"step": step + 1, "predicted_rainfall": next_value})
        last_window = np.append(last_window[1:], next_scaled).astype(np.float32)
    lstm_model = {"network": network, "time_column": time_column, "rainfall_column": rainfall_column}
    return {"source": "Calculated from Uploaded Dataset", "status": "trained", "model": "LSTM", "units": request.units, "optimizer": "Adam", "epochs": request.epochs, "lookback": lookback, "training_observations": int(len(x_train)), "validation_observations": int(len(x_val)), "training_seconds": training_seconds, "loss_history": history, "validation": [{"actual": float(actual[index]), "predicted": float(val_prediction_values[index])} for index in range(len(actual))], "forecast": forecast, "time_column": time_column, "rainfall_column": rainfall_column, "warning": "Rainfall forecast only; this is not a flood-severity forecast. Validate time cadence and external conditions before use."}


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    model = trained_models.get(request.model)
    if not model:
        raise HTTPException(409, f"Model '{request.model}' has not been trained in this session.")
    input_values = {feature: request.features.get(feature) for feature in model["features"]}
    sample = pd.DataFrame([{key: value for key, value in input_values.items()}], columns=model["features"])
    try:
        predicted_index = int(model["pipeline"].predict(sample)[0])
        probabilities = model["pipeline"].predict_proba(sample)[0]
    except Exception as error:
        raise HTTPException(422, f"Prediction failed for the supplied feature values: {error}") from error
    label = str(model["encoder"].inverse_transform([predicted_index])[0])
    probability = float(np.max(probabilities)) if probabilities is not None else None
    risk, action = _risk_label(label)
    result = _save_prediction(request.model, input_values, label, probability, risk, request)
    result.update({"risk_action": action, "future_year": request.future_year, "state": request.state, "district": request.district, "warning_disclaimer": "Research-generated decision support, not an official government warning."})
    return result


@app.post("/api/future-prediction")
def future_prediction(request: PredictRequest) -> dict[str, Any]:
    if request.future_year is None:
        raise HTTPException(422, "Provide a future year for the scenario.")
    if request.future_year <= datetime.now(timezone.utc).year:
        raise HTTPException(422, "Future scenario year must be later than the current year.")
    data = _current()
    year_feature = next((feature for feature in trained_models.get(request.model, {}).get("features", []) if "year" in _norm(feature)), None)
    if year_feature:
        request.features[year_feature] = request.future_year
    result = predict(request)
    result["mode"] = "historical_model_scenario"
    result["methodology"] = f"The requested year was supplied to the model's observed '{year_feature}' feature. A historical classifier applies learned split rules; it does not extrapolate future annual risk." if year_feature else "The trained historical classifier evaluates supplied feature values. The requested year is metadata only; it is not extrapolated because the model has no year feature."
    result["time_series_forecast_available"] = _time_series_status(data)["suitable"]
    return result


@app.get("/api/research-results")
def research_results() -> dict[str, Any]:
    return {"source": "Reported in Research Paper", **PAPER_RESULTS}


@app.get("/api/model-comparison")
def model_comparison() -> dict[str, Any]:
    return {"source": "Calculated from Uploaded Dataset", "models": [
        {"model": item["model"], **item["metrics"], "timing": item["timing"]} for item in trained_results.values()
    ]}


@app.get("/api/prediction-history")
def prediction_history(search: str = "", model: str | None = None, limit: int = Query(default=500, ge=1, le=5000)) -> dict[str, Any]:
    conditions = " WHERE 1=1"
    values: list[Any] = []
    if model:
        conditions += " AND model = ?"
        values.append(model)
    if search:
        conditions += " AND (predicted_severity LIKE ? OR state LIKE ? OR district LIKE ?)"
        term = f"%{search}%"
        values.extend([term, term, term])
    with _db() as connection:
        total = connection.execute(f"SELECT COUNT(*) FROM predictions{conditions}", values).fetchone()[0]
        rows = connection.execute(f"SELECT * FROM predictions{conditions} ORDER BY id DESC LIMIT ?", [*values, limit]).fetchall()
    return {"total": total, "records": [{"id": row["id"], "timestamp": row["timestamp"], "model": row["model"], "inputs": json.loads(row["input_json"]), "predicted_severity": row["predicted_severity"], "probability": row["probability"], "risk_level": row["risk_level"], "future_year": row["future_year"], "state": row["state"], "district": row["district"]} for row in rows]}


@app.delete("/api/prediction-history/{prediction_id}")
def delete_prediction(prediction_id: int) -> dict[str, bool]:
    with _db() as connection:
        cursor = connection.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    if cursor.rowcount == 0:
        raise HTTPException(404, "Prediction not found.")
    return {"deleted": True}


@app.get("/api/prediction-history.csv")
def export_prediction_history() -> Response:
    with _db() as connection:
        rows = connection.execute("SELECT timestamp,model,input_json,predicted_severity,probability,risk_level,future_year,state,district FROM predictions ORDER BY id DESC").fetchall()
    records = [{"timestamp": row[0], "model": row[1], "inputs": row[2], "predicted_severity": row[3], "probability": row[4], "risk_level": row[5], "future_year": row[6], "state": row[7], "district": row[8]} for row in rows]
    stream = io.StringIO()
    pd.DataFrame(records).to_csv(stream, index=False)
    return Response(stream.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="prediction-history.csv"'})


@app.get("/api/research-results/{section}")
def research_section(section: str) -> dict[str, Any]:
    if section not in PAPER_RESULTS:
        raise HTTPException(404, "Unknown research-results section.")
    return {"source": "Reported in Research Paper", "section": section, "data": PAPER_RESULTS[section]}


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
