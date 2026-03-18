#Imports 

import sqlite3
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
)
import mlflow
import os
import json
from datetime import datetime
import time

#Load predictions from the database 

def load_predictions(db_path: str = "predictions.db") -> pd.DataFrame:
    if not os.path.exists(db_path):
        print("No predictions database found yet.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY timestamp DESC",
        conn
    )
    conn.close()

    print(f"Loaded {len(df)} predictions from database.")
    return df

#Create a baseline 

def create_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    The baseline is the first 100 predictions our model made.
    We assume the model was healthy then — this becomes our reference point.
    """
    if len(df) < 100:
        print("Not enough predictions for baseline (need at least 100).")
        return pd.DataFrame()

    baseline = df.tail(100).copy()
    baseline = baseline[["confidence", "label"]]

    # Convert label to numeric for drift detection
    baseline["label_numeric"] = baseline["label"].map(
        {"positive": 1, "negative": 0}
    )

    return baseline

#Check the drift 

def check_drift(current_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict:
    if len(current_df) < 50:
        print("Not enough current predictions to check drift (need 50).")
        return {"drift_detected": False, "reason": "insufficient_data"}

    # Prepare current data
    current = current_df.head(50).copy()
    current = current[["confidence", "label"]]
    current["label_numeric"] = current["label"].map(
        {"positive": 1, "negative": 0}
    )

    # Prepare baseline for comparison
    baseline = baseline_df[["confidence", "label_numeric"]].copy()
    current_compare = current[["confidence", "label_numeric"]].copy()

    # Run Evidently drift report
    report = Report(metrics=[
        DatasetDriftMetric(),
        ColumnDriftMetric(column_name="confidence"),
        ColumnDriftMetric(column_name="label_numeric"),
    ])

    report.run(
        reference_data=baseline,
        current_data=current_compare
    )

    # Extract results
    report_dict = report.as_dict()
    dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
    confidence_drift = report_dict["metrics"][1]["result"]["drift_detected"]

    # Save the HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"./reports/drift_report_{timestamp}.html"
    os.makedirs("./reports", exist_ok=True)
    report.save_html(report_path)
    print(f"Drift report saved to {report_path}")

    result = {
        "drift_detected": dataset_drift,
        "confidence_drift": confidence_drift,
        "timestamp": datetime.now().isoformat(),
        "report_path": report_path
    }

    print(f"Drift detected: {dataset_drift}")
    print(f"Confidence drift: {confidence_drift}")

    return result

#Trigger retaining 

def trigger_retraining():
    print("Drift detected! Triggering retraining...")

    # Log the retraining event to MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")

    with mlflow.start_run(run_name="auto-retrain-trigger"):
        mlflow.log_param("trigger", "drift_detected")
        mlflow.log_param("timestamp", datetime.now().isoformat())

    # In production this would call an orchestration tool like Airflow
    # For now we write a trigger file that Docker Compose watches
    with open("./retrain_trigger.flag", "w") as f:
        f.write(datetime.now().isoformat())

    print("Retraining trigger saved. Training container will pick this up.")
    
#The main monitoring loop 

def run_monitoring(interval_seconds: int = 60):
    """
    Runs monitoring on a schedule.
    Every 60 seconds, checks for drift.
    """
    print(f"Starting monitoring loop (checking every {interval_seconds}s)...")
    mlflow.set_tracking_uri("http://mlflow:5000")

    while True:
        print(f"\n--- Monitoring check at {datetime.now().isoformat()} ---")

        try:
            # Load all predictions
            df = load_predictions()

            if df.empty:
                print("No predictions yet. Waiting...")
                time.sleep(interval_seconds)
                continue

            # Create baseline from oldest predictions
            baseline_df = create_baseline(df)

            if baseline_df.empty:
                print("Not enough data for baseline yet. Waiting...")
                time.sleep(interval_seconds)
                continue

            # Check for drift
            drift_result = check_drift(df, baseline_df)

            # Save drift result as JSON
            with open("./reports/latest_drift.json", "w") as f:
                json.dump(drift_result, f, indent=2)

            # Trigger retraining if drift detected
            if drift_result.get("drift_detected"):
                trigger_retraining()
            else:
                print("No drift detected. Model is healthy.")

        except Exception as e:
            print(f"Monitoring error: {e}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    run_monitoring(interval_seconds=60)
    
