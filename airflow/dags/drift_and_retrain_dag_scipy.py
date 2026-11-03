# Standard library datetime: establishes the initial reference date for scheduling[cite: 1]
from datetime import datetime

# Standard library timedelta: provides time durations for intervals or retries[cite: 1]
from datetime import timedelta

# Built-in JSON parser: deserializes stringified JSON feature lists pulled from Postgres[cite: 1]
import json

# PostgreSQL driver: handles raw SQL connections to the prediction log store[cite: 1]
import psycopg2

# Pandas library: tabular data structures for organizing extracted database rows[cite: 1]
import pandas as pd

# NumPy library: numerical tools for array flattening and synthetic baseline generation[cite: 1]
import numpy as np

# Core Airflow DAG class: context manager defining the pipeline structure[cite: 1]
from airflow import DAG

# PythonOperator: Airflow worker operator that wraps Python functions into DAG task nodes[cite: 1]
from airflow.operators.python import PythonOperator

# Ray SDK client: submits distributed Python jobs to the Ray Head service[cite: 1]
from ray.job_submission import JobSubmissionClient

# Baseline task properties applied across all operators in this file[cite: 1]
default_args = {
    # Name of the pipeline owner visible in the Airflow dashboard[cite: 1]
    'owner': 'mlops',
    # Zero retries: fails immediately if an uncaught exception is raised[cite: 1]
    'retries': 0,
    # The earliest possible execution boundary for this workflow[cite: 1]
    'start_date': datetime(2026, 1, 1),
}

# Task 1 Callable: reads the recent feature window and tests for distribution shifts[cite: 1]
def check_drift_with_scipy(**context):
    """Fetches recent prediction logs and evaluates univariate distribution shifts via SciPy KS test."""
    # Connect directly to the serving database container[cite: 1]
    conn = psycopg2.connect("postgresql://mlops_admin:mlops_password123@postgres:5432/serving_db")[cite: 1]
    # Fetch up to 500 of the newest inference feature vectors[cite: 1]
    df_curr = pd.read_sql("SELECT features FROM prediction_logs ORDER BY timestamp DESC LIMIT 500", conn)[cite: 1]
    # Close database socket immediately after querying[cite: 1]
    conn.close()[cite: 1]

    # Guard clause: ensure sufficient statistical power before running the test[cite: 1]
    if len(df_curr) < 50:[cite: 1]
        # Log notice to Airflow task logs[cite: 1]
        print("Not enough samples to compute drift.")[cite: 1]
        # Return False to halt any downstream retraining[cite: 1]
        return False[cite: 1]

    # Convert the JSON array strings into a dense 2D NumPy array[cite: 1]
    current_matrix = np.array([json.loads(r) for r in df_curr['features']])[cite: 1]
    
    # Generate a synthetic Gaussian reference baseline matching the matrix dimensions[cite: 1]
    reference_matrix = np.random.normal(0.0, 1.0, size=current_matrix.shape)[cite: 1]

    # Import the two-sample Kolmogorov-Smirnov test from SciPy[cite: 1]
    from scipy.stats import ks_2samp[cite: 1]
    
    # Flatten both matrices to test aggregate distribution equality[cite: 1]
    stat, p_value = ks_2samp(reference_matrix.flatten(), current_matrix.flatten())[cite: 1]
    
    # Log test statistic and p-value to the execution log[cite: 1]
    print(f"KS-Test Statistic: {stat}, P-Value: {p_value}")[cite: 1]

    # Consider drift detected if the p-value is strictly below 0.05[cite: 1]
    drift_detected = bool(p_value < 0.05)[cite: 1]
    
    # Return boolean result; pushed to XCom for downstream evaluation[cite: 1]
    return drift_detected[cite: 1]

# Task 2 Callable: inspects Task 1 output and optionally launches training on Ray[cite: 1]
def trigger_ray_retrain(**context):
    """Retrieves XCom output from drift check and launches a Ray job if distributions shifted."""
    # Retrieve Task 1's boolean return value from Airflow XCom[cite: 1]
    drift_detected = context['ti'].xcom_pull(task_ids='check_drift')[cite: 1]
    
    # If no drift occurred, skip retraining[cite: 1]
    if not drift_detected:[cite: 1]
        print("No significant drift detected. Retraining skipped.")[cite: 1]
        return[cite: 1]

    # Log retraining submission[cite: 1]
    print("Drift confirmed. Submitting distributed training job to Ray head...")[cite: 1]
    
    # Point the Ray client to the head node REST endpoint[cite: 1]
    client = JobSubmissionClient("http://ray-head:8265")[cite: 1]
    
    # Submit the training script to run inside the Ray worker cluster[cite: 1]
    job_id = client.submit_job(
        # Entrypoint script path inside the container[cite: 1]
        entrypoint="python /home/ray/training/train.py",[cite: 1]
        # Runtime dependencies installed in the Ray worker environment[cite: 1]
        runtime_env={"pip": ["mlflow==2.13.0", "scikit-learn==1.5.0", "boto3==1.34.120"]}[cite: 1]
    )
    
    # Log assigned Ray job ID[cite: 1]
    print(f"Submitted Ray Job ID: {job_id}")[cite: 1]

# ==========================================
# AIRFLOW DAG DEFINITION & OPERATOR LINKING
# ==========================================

# Open the Airflow DAG context manager explicitly
with DAG(
    # Unique identifier used in the Airflow web UI
    dag_id='scipy_drift_monitor',
    # Default settings dictionary defined above[cite: 1]
    default_args=default_args,[cite: 1]
    # Execution interval: runs automatically every hour[cite: 1]
    schedule_interval='@hourly',[cite: 1]
    # Disables backfill runs for missed historical intervals[cite: 1]
    catchup=False[cite: 1]
) as dag:

    # Task 1: Check feature drift with SciPy[cite: 1]
    drift_task = PythonOperator(
        # Task identifier shown in the graph view[cite: 1]
        task_id='check_drift',[cite: 1]
        # Target Python callable to execute[cite: 1]
        python_callable=check_drift_with_scipy,
        # Grants access to Airflow context and XCom[cite: 1]
        provide_context=True[cite: 1]
    )

    # Task 2: Trigger retrain job on Ray[cite: 1]
    retrain_task = PythonOperator(
        # Task identifier shown in the graph view[cite: 1]
        task_id='trigger_retrain',[cite: 1]
        # Target Python callable to execute[cite: 1]
        python_callable=trigger_ray_retrain,[cite: 1]
        # Grants access to Airflow context and XCom[cite: 1]
        provide_context=True[cite: 1]
    )

    # Strict sequential dependency: check_drift must complete before trigger_retrain executes[cite: 1]
    drift_task >> retrain_task[cite: 1]