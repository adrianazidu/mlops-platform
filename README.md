# Local End-to-End MLOps Platform (Docker Compose)

An enterprise-pattern MLOps platform running locally via Docker Compose on Linux / WSL 2. 
It supports low-latency model inference, distributed training on Ray, experiment tracking with MLflow, object storage via MinIO, automated drift monitoring with Evidently AI, and operational observability via Prometheus & Grafana.

---

## Architecture Overview

[ Client / Ingestion ]
│
▼
[ FastAPI Serving (:8000) ] ────► [ Prometheus (:9090) ] ──► [ Grafana (:3000) ]
│ (Async Logs)
▼
[ PostgreSQL (:5432) ] ◄────────┐
│                      │
▼                      │
[ Airflow (:8080) ] ────────────┤
(Hourly DAG)                 │
│                      │
├─► [ Evidently AI ] ──┘ (Reads inference logs)
│       │
│       └─► Generates Drift Reports ──► [ MinIO Storage (:9000/:9001) ]
│
Drift > Threshold?
├─ No  ──► Pass
└─ Yes ──► Submit Retrain Job ──► [ Ray Cluster (:8265) ]
│
└─► Registers Model ──► [ MLflow (:5000) ]