# Standard library datetime: used to define DAG schedule start dates and intervals
from datetime import datetime

# Built-in JSON parser: deserializes the JSONB feature lists extracted from PostgreSQL
import json

# PostgreSQL client driver: enables direct SQL connections to query the inference database
import psycopg2

# Pandas library: structures raw database records into tabular DataFrames for analysis
import pandas as pd

# NumPy: used to generate numerical reference arrays and manipulate matrix shapes
import numpy as np

# Core Airflow DAG class: represents the workflow graph holding scheduled tasks
from airflow import DAG

# PythonOperator: Airflow operator that wraps standard Python callables as executable graph nodes
from airflow.operators.python import PythonOperator

# Ray SDK client: connects over HTTP to the Ray Head node to submit remote execution jobs
from ray.job_submission import JobSubmissionClient

# Evidently AI Report: central engine for executing evaluation suites across reference and production datasets
from evidently.report import Report

# Evidently Data Drift Preset: pre-packaged suite of statistical drift tests tailored for tabular data
from evidently.metric_preset import DataDriftPreset

# Base configuration dictionary applied across all tasks declared within the DAG
default_args = {
    # Label identifying the team or process owner in the Airflow web interface
    'owner': 'mlops',
    # Maximum execution retries if a task raises an unhandled exception (0 disables retries)
    'retries': 0,
    # Execution reference start timestamp (Airflow will not schedule runs before this date)
    'start_date': datetime(2026, 1, 1),
}

# Task callable responsible for extracting logs, constructing baselines, and measuring distribution shifts
def check_drift_with_evidently(**context):
    """Extracts recent production feature logs and checks for drift using Evidently AI."""
    # Open an active connection to the PostgreSQL database housing serving logs
    conn = psycopg2.connect("postgresql://mlops_admin:mlops_password123@postgres:5432/serving_db")
    # Query up to the last 500 prediction payloads ordered chronologically descending
    df_curr = pd.read_sql("SELECT features FROM prediction_logs ORDER BY timestamp DESC LIMIT 500", conn)
    # Terminate the database connection to free socket and pool resources
    conn.close()

    # Guard clause: ensure sufficient statistical power before running distribution tests
    if len(df_curr) < 20:
        # Log notice to Airflow task logs
        print("[Evidently] Less than 20 rows found. Insufficient sample size for drift test.")
        # Return False to signify no drift / abort downstream retraining
        return False

    # Extract JSON strings from the database column and convert them into Python float arrays
    current_features = [json.loads(row) for row in df_curr['features']]
    # Dynamically generate column headers (f_0, f_1, ...) based on the incoming vector dimension
    col_names = [f"f_{i}" for i in range(len(current_features[0]))]
    # Rebuild features into a structured Pandas DataFrame representing production inputs
    current_df = pd.DataFrame(current_features, columns=col_names)

    # Fix pseudo-random seed to guarantee reproducible reference baseline distributions
    np.random.seed(42)
    # Simulate historical validation dataset sampled from a standard Gaussian N(0, 1)
    reference_df = pd.DataFrame(
        np.random.normal(0.0, 1.0, size=(200, len(col_names))),
        columns=col_names
    )

    # Initialize Evidently Report configured with standard automated data drift evaluation presets
    report = Report(metrics=[DataDriftPreset()])
    # Execute the comparison between historical baseline (reference) and current production inputs
    report.run(reference_data=reference_df, current_data=current_df)
    # Export full interactive HTML report visualizer to the persistent logs directory
    report.save_html("/opt/airflow/logs/evidently_drift_report.html")

    # Serialize test metrics into an accessible nested Python dictionary
    result_dict = report.as_dict()
    # Extract binary verdict indicating if enough individual features crossed significance thresholds
    dataset_drift = result_dict["metrics"][0]["result"]["dataset_drift"]
    # Retrieve the exact proportion of columns flagged as drifted (e.g., 0.40 = 40%)
    drift_share = result_dict["metrics"][0]["result"]["share_of_drifted_columns"]

    # Log summary of statistical evaluation to standard output for Airflow UI inspection
    print(f"[Evidently] Drifted Column Share: {drift_share:.2%}, Drift Alert Triggered: {dataset_drift}")
    # Return the boolean decision; Airflow automatically pushes this value to XCom
    return bool(dataset_drift)

# Task callable that evaluates drift results and submits an automated training run to Ray
def trigger_ray_retrain(**context):
    """Submits a training job to the Ray cluster head node if drift was detected."""
    # Retrieve the boolean return value generated by 'check_drift' from Airflow's XCom storage
    drift_detected = context['ti'].xcom_pull(task_ids='check_drift')
    # If drift was not triggered, terminate task without firing training workloads
    if not drift_detected:
        # Print informational notice to task logs
        print("[Orchestrator] Data distributions nominal. Retraining omitted.")
        # Exit cleanly
        return

    # Log confirmation that drift was flagged and the cluster job is dispatching
    print("[Orchestrator] Significant drift detected. Submitting job to Ray Head (8265)...")
    # Initialize the client pointing to the Ray Head node REST submission port
    client = JobSubmissionClient("http://ray-head:8265")
    # Submit asynchronous distributed job request across the container cluster
    job_id = client.submit_job(
        # Command executed inside the Ray container runtime
        entrypoint="python /home/ray/training/train.py",
        # Isolate job-specific pip dependencies dynamically inside the worker environment
        runtime_env={"pip": ["mlflow==2.13.0", "scikit-learn==1.5.0", "boto3==1.34.120"]}
    )
    # Output the newly assigned Ray execution job ID for cross-system tracking
    print(f"[Orchestrator] Job successfully accepted by Ray. Job ID: {job_id}")

# Instantiate the DAG workflow using context manager syntax
with DAG(
    # Unique identifier used in the Airflow dashboard to track this pipeline
    dag_id='evidently_drift_monitor',
    # Apply default parameters defined above
    default_args=default_args,
    # Cron schedule specifying execution frequency (runs once every hour)
    schedule_interval='@hourly',
    # Prevent backfilling historical execution periods that were missed
    catchup=False
) as dag:

    # Define first task: execute Evidently AI drift detection on prediction logs
    drift_task = PythonOperator(
        # Internal task identifier within the DAG
        task_id='check_drift',
        # Target Python function to execute
        python_callable=check_drift_with_evidently,
        # Injects execution context (including XCom 'ti' handles) into keyword arguments
        provide_context=True
    )

    # Define second task: evaluate XCom result and trigger Ray retraining job
    retrain_task = PythonOperator(
        # Internal task identifier within the DAG
        task_id='trigger_retrain',
        # Target Python function to execute
        python_callable=trigger_ray_retrain,
        # Injects execution context to read results from upstream tasks
        provide_context=True
    )

    # Bitshift operator setting strict sequential dependency: check_drift must complete before trigger_retrain runs
    drift_task >> retrain_task