# Operating system interfaces: reads environment variables such as MLFLOW_TRACKING_URI from the container
import os

# MLflow SDK: logs parameters, performance metrics, and binary model artifacts to the tracking server
import mlflow

# Scikit-learn LogisticRegression: linear classification model trained on feature vectors
from sklearn.linear_model import LogisticRegression

# NumPy: generates synthetic training matrices and handles numerical array transformations
import numpy as np

# Point MLflow client to the central tracking container (defaults to http://mlflow:5000 inside the Docker network)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Define or locate the experiment namespace under which this training run and its runs will be categorized
mlflow.set_experiment("document_classification")

# Print status to stdout so the job submission progress can be monitored via Ray Dashboard or CLI logs
print("[Ray Training] Starting training routine...")

# Open an active MLflow tracking context manager; automatically initializes a run ID and records timestamps
with mlflow.start_run():
    # Generate a synthetic feature matrix X containing 300 samples and 10 features sampled from standard normal distributions
    X = np.random.randn(300, 10)
    
    # Generate binary ground truth targets y: 1 if the sum of row features is positive, 0 otherwise
    y = (np.sum(X, axis=1) > 0).astype(int)
    
    # Instantiate the scikit-learn classifier with default L2 regularization and L-BFGS solver
    model = LogisticRegression()
    
    # Train the decision boundaries on the feature matrix X and ground truth label vector y
    model.fit(X, y)
    
    # Calculate classification accuracy on the training dataset (fraction of correct predictions)
    train_acc = float(model.score(X, y))
    
    # Send the calculated accuracy metric to the central MLflow server to be stored in PostgreSQL
    mlflow.log_metric("train_accuracy", train_acc)
    
    # Serialize the scikit-learn model object, upload weights to MinIO (s3://mlflow-artifacts), and register model in MLflow Registry
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", 
        registered_model_name="DocumentClassificationModel"
    )
    
    # Print completion summary including final accuracy metric to container execution logs
    print(f"[Ray Training] Run finished. Accuracy: {train_acc:.4f}. Model registered to MLflow.")