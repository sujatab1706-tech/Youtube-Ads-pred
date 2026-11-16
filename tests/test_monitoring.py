"""End-to-end check: simulated drift must be caught by monitoring and fixed by retraining."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True)


def test_drift_is_detected_and_retraining_helps(tmp_path):
    data = tmp_path / "ads.csv"
    run("src/generate_data.py", "--rows", "5000", "--out", str(data))
    models = tmp_path / "models"
    shutil.copytree(ROOT / "models", models)

    healthy, drifted = tmp_path / "healthy", tmp_path / "drifted"
    run("src/simulate_traffic.py", "--scenario", "normal", "--rows", "800", "--log-dir", str(healthy))
    run("src/simulate_traffic.py", "--scenario", "drift", "--rows", "800", "--log-dir", str(drifted))

    for logs in (healthy, drifted):
        run("src/monitor.py", "--data", str(data), "--model", str(models / "model.joblib"),
            "--log-dir", str(logs), "--out", str(logs / "reports"))

    healthy_summary = json.loads((healthy / "reports" / "monitoring_summary.json").read_text())
    drift_summary = json.loads((drifted / "reports" / "monitoring_summary.json").read_text())
    assert healthy_summary["retrain_needed"] is False
    assert drift_summary["retrain_needed"] is True
    assert "device" in drift_summary["drifted_features"]

    run("src/retrain.py", "--data", str(data), "--model", str(models / "model.joblib"), "--log-dir", str(drifted))
    report = json.loads((models / "retrain_report.json").read_text())
    assert report["promoted"] is True
    assert (models / "model_previous.joblib").exists()
