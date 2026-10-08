from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from backend import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATABASE", tmp_path / "predictions.sqlite3")
    main.frame = None
    main.processed_frame = None
    main.original_filename = None
    main.preprocessing_config = {}
    main.trained_models = {}
    main.trained_results = {}
    main.lstm_model = None
    with TestClient(main.app) as test_client:
        yield test_client


def make_csv() -> bytes:
    rows = ["Duration(D),Main Cause,State,Latitude,Longitude,Area Affected,Human fatalities,Human Displaced,Severity"]
    for index in range(40):
        severity = "S2" if index % 2 else "S1"
        duration = index + (20 if severity == "S2" else 0)
        rows.append(f'{duration},"Rain, monsoon",Assam,{24 + index / 100},{90 + index / 100},{1000 + index},2,100,{severity}')
    rows.append(rows[-1])
    return "\n".join(rows).encode()


def load_csv(client: TestClient) -> dict:
    response = client.post("/api/dataset/upload", files={"file": ("test-inventory.csv", io.BytesIO(make_csv()), "text/csv")})
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_reports_dataset_facts_and_target_distribution(client):
    info = load_csv(client)
    assert info["rows"] == 41
    assert info["columns_count"] == 9
    assert info["duplicate_rows"] == 1
    assert info["target_candidate"] == "Severity"
    assert {item["label"]: item["count"] for item in info["target_distribution"]} == {"S1": 20, "S2": 21}
    assert info["geographic_columns"]["latitude"] == "Latitude"
    assert info["time_series"]["suitable"] is False


def test_preview_search_and_python_preprocessing(client):
    load_csv(client)
    preview = client.get("/api/dataset/preview?search=monsoon&limit=10").json()
    assert preview["filtered_rows"] == 41
    transformed = client.post("/api/preprocess", json={"target": "Severity", "drop_duplicates": True, "missing_strategy": "impute"})
    assert transformed.status_code == 200, transformed.text
    assert transformed.json()["processed"]["rows"] == 40
    assert main.frame is not None and len(main.frame) == 41


def test_random_forest_metrics_prediction_and_history_are_real(client):
    load_csv(client)
    trained = client.post("/api/train/random-forest", json={"target": "Severity", "features": ["Duration(D)", "Main Cause", "State"], "n_estimators": 20, "test_size": 0.25})
    assert trained.status_code == 200, trained.text
    result = trained.json()
    assert result["source"] == "Calculated from Uploaded Dataset"
    assert result["split"]["train_rows"] + result["split"]["test_rows"] == 40
    assert sum(sum(row) for row in result["confusion_matrix"]["values"]) == result["split"]["test_rows"]
    assert len(result["predictions"]) == result["split"]["test_rows"]
    prediction = client.post("/api/predict", json={"model": "random-forest", "features": {"Duration(D)": 35, "Main Cause": "Rain, monsoon", "State": "Assam"}})
    assert prediction.status_code == 200, prediction.text
    history = client.get("/api/prediction-history").json()["records"]
    assert len(history) == 1
    assert history[0]["predicted_severity"] == prediction.json()["predicted_severity"]


def test_lstm_refuses_non_chronological_data(client):
    load_csv(client)
    response = client.post("/api/train/lstm", json={})
    assert response.status_code == 422
    assert "does not contain sufficient chronological" in response.json()["detail"]


def test_research_paper_results_are_separately_labeled(client):
    response = client.get("/api/research-results")
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "Reported in Research Paper"
    assert payload["classification"][1]["accuracy"] == 65.82
