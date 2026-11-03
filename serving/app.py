# Operating system interfaces: used to retrieve environment variables (e.g., POSTGRES_URI)
import os

# Standard JSON serializer: transforms Python lists/dictionaries into JSON strings for PostgreSQL storage
import json

# Time library: measures wall-clock execution time to calculate inference latency in milliseconds
import time

# PostgreSQL database adapter: connects and executes SQL queries against the serving_db instance
import psycopg2

# FastAPI framework: creates the web application, routing endpoints, and dependency management
# BackgroundTasks: schedules non-blocking background workers after returning the HTTP response
from fastapi import FastAPI, BackgroundTasks

# Pydantic BaseModel: enforces strict schema definition, typing, and validation on incoming HTTP JSON payloads
from pydantic import BaseModel

# Prometheus instrumentator: exposes standard HTTP web metrics (throughput, duration buckets, status codes)
from prometheus_fastapi_instrumentator import Instrumentator

# Prometheus Gauge metric: represents a numerical value that can arbitrarily go up and down
from prometheus_client import Gauge

# NumPy: high-performance linear algebra library for tensor/vector operations and mathematical transformations
import numpy as np

# Instantiate the primary FastAPI application object with a descriptive documentation title
app = FastAPI(title="Document Inference Service")

# Initialize a custom Prometheus Gauge metric to expose model prediction entropy to the /metrics endpoint
# Shannon entropy serves as an unsupervised proxy for predictive uncertainty / out-of-distribution detection
INFERENCE_ENTROPY = Gauge(
    "model_prediction_entropy", 
    "Shannon entropy of current model predictions (task degradation proxy)"
)

# Automatically attach Prometheus hooks to FastAPI request lifecycle and bind the scrape route at /metrics
Instrumentator().instrument(app).expose(app)

# Read the PostgreSQL connection string passed from the container environment variables defined in .env
POSTGRES_URI = os.getenv("POSTGRES_URI")

# Define a bootstrap function to guarantee the inference audit table exists in PostgreSQL before serving traffic
def init_db():
    # Establish a synchronous TCP connection to the PostgreSQL serving database
    conn = psycopg2.connect(POSTGRES_URI)
    # Create a database cursor to execute SQL statements within the active transaction
    cur = conn.cursor()
    # Execute a DDL statement creating the prediction_logs table if it does not already exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            doc_id VARCHAR(64),
            features JSONB,
            prediction FLOAT,
            entropy FLOAT,
            latency_ms FLOAT
        );
    """)
    # Commit the transaction to disk to persist table creation
    conn.commit()
    # Close the cursor resource
    cur.close()
    # Close the database connection pool socket
    conn.close()

# Event hook triggered automatically by FastAPI right before the server begins accepting incoming connections
@app.on_event("startup")
def startup_event():
    # Pause execution briefly to give PostgreSQL time to accept TCP connections during cold container boot
    time.sleep(2)
    # Invoke the database schema initialization function
    init_db()

# Worker function executed outside the client's HTTP request-response loop via BackgroundTasks
def log_to_postgres(doc_id: str, features: list, prediction: float, entropy: float, latency: float):
    # Wrap database I/O in a try block to ensure logging errors never crash the main application process
    try:
        # Open a distinct database connection for this worker thread
        conn = psycopg2.connect(POSTGRES_URI)
        # Allocate a new execution cursor
        cur = conn.cursor()
        # Parameterize SQL INSERT statement to safely store the inference payload and prevent SQL injection
        cur.execute(
            "INSERT INTO prediction_logs (doc_id, features, prediction, entropy, latency_ms) VALUES (%s, %s, %s, %s, %s)",
            (doc_id, json.dumps(features), prediction, entropy, latency)
        )
        # Commit the transaction so Evidently AI or Airflow can query it immediately
        conn.commit()
        # Release cursor resources
        cur.close()
        # Close connection back to the database
        conn.close()
    # Catch any connection timeouts, authentication failures, or transient SQL errors
    except Exception as e:
        # Print the error traceback to stdout for docker compose logs collection without failing the client request
        print(f"[Logging Error] Could not write prediction to database: {e}")

# Define the expected JSON request body schema sent by clients querying the model
class DocumentRequest(BaseModel):
    # Unique identifier string representing the document or request payload
    doc_id: str
    # Ordered array of floating point numbers representing the pre-computed document feature embeddings
    feature_vector: list[float]

# Register a POST route at /predict that consumes the DocumentRequest payload and accepts FastAPI BackgroundTasks
@app.post("/predict")
def predict(payload: DocumentRequest, background_tasks: BackgroundTasks):
    # Record start timestamp with high precision to benchmark end-to-end inference execution time
    t0 = time.time()
    
    # Cast incoming Python list of floats into a contiguous 1D NumPy array for fast vectorized math
    feat_arr = np.array(payload.feature_vector)
    
    # Compute simulated logistic sigmoid prediction: 1 / (1 + exp(-w^T * x)) across feature vector sum
    prob = 1.0 / (1.0 + np.exp(-np.sum(feat_arr)))
    
    # Construct binary probability distribution array [p, 1 - p] and clip to prevent log(0) numeric underflows
    p = np.clip([prob, 1.0 - prob], 1e-12, 1.0)
    
    # Calculate Shannon Entropy: H(p) = -sum(p * ln(p)); returns uncertainty value in nats
    entropy = float(-np.sum(p * np.log(p)))
    
    # Set the Prometheus gauge to the newly computed entropy value for real-time Grafana dashboard polling
    INFERENCE_ENTROPY.set(entropy)

    # Compute total elapsed inference latency and convert seconds to milliseconds
    latency = (time.time() - t0) * 1000.0

    # Offload the database write operation to the background task executor so the HTTP client receives an immediate response
    background_tasks.add_task(
        log_to_postgres, 
        payload.doc_id, 
        payload.feature_vector, 
        float(prob), 
        entropy, 
        latency
    )

    # Return the prediction response payload back to the calling client
    return {"doc_id": payload.doc_id, "prob": float(prob), "entropy": entropy}