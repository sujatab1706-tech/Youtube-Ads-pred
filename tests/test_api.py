"""Tests for the prediction API."""

from fastapi.testclient import TestClient

from app.main import app

AD = {
    "category": "Gaming", "device": "Mobile", "ad_format": "In-feed", "age_group": "18-24",
    "hour": 20, "day_of_week": 5, "ad_length_sec": 15, "daily_budget_usd": 150.0,
    "impressions": 40000, "channel_subscribers": 250000, "video_watch_pct": 60.0,
    "thumbnail_quality": 4,
}


def client():
    return TestClient(app)


def test_health():
    with client() as c:
        r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_sensible_clicks():
    with client() as c:
        r = c.post("/predict", json=AD)
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["predicted_clicks"] < AD["impressions"]
    assert 0 < body["predicted_ctr_pct"] < 10


def test_optional_fields_can_be_missing():
    ad = {k: v for k, v in AD.items() if k not in ("channel_subscribers", "video_watch_pct")}
    with client() as c:
        r = c.post("/predict", json=ad)
    assert r.status_code == 200


def test_invalid_input_is_rejected():
    with client() as c:
        assert c.post("/predict", json={**AD, "device": "Smartwatch"}).status_code == 422
        assert c.post("/predict", json={**AD, "hour": 25}).status_code == 422
        assert c.post("/predict", json={**AD, "impressions": 0}).status_code == 422


def test_more_impressions_means_more_clicks():
    with client() as c:
        small = c.post("/predict", json={**AD, "impressions": 10000}).json()["predicted_clicks"]
        big = c.post("/predict", json={**AD, "impressions": 100000}).json()["predicted_clicks"]
    assert big > small


def test_tv_gets_fewer_clicks_than_mobile():
    with client() as c:
        mobile = c.post("/predict", json=AD).json()["predicted_clicks"]
        tv = c.post("/predict", json={**AD, "device": "TV"}).json()["predicted_clicks"]
    assert tv < mobile


def test_batch():
    with client() as c:
        r = c.post("/predict/batch", json={"ads": [AD, {**AD, "device": "TV"}]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_model_info():
    with client() as c:
        body = c.get("/model-info").json()
    assert body["model_type"] == "linear_log" and body["test_metrics"]["r2"] > 0.9


def test_predictions_and_feedback_are_logged():
    import json

    from app.main import FEEDBACK_LOG, PREDICTION_LOG

    with client() as c:
        pid = c.post("/predict", json=AD).json()["prediction_id"]
        assert c.post("/feedback", json={"prediction_id": pid, "actual_clicks": 1200}).status_code == 200
    logged = [json.loads(line) for line in PREDICTION_LOG.read_text().splitlines()]
    assert any(r["prediction_id"] == pid and r["device"] == "Mobile" for r in logged)
    feedback = [json.loads(line) for line in FEEDBACK_LOG.read_text().splitlines()]
    assert feedback[-1] == {**feedback[-1], "prediction_id": pid, "actual_clicks": 1200}
