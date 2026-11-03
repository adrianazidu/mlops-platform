# Python runtime optimized for inference
FROM python:3.10-slim

WORKDIR /app

# Core runtime dependencies: web framework, metrics exporter, DB driver, math
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    prometheus-fastapi-instrumentator==7.0.0 \
    prometheus-client==0.20.0 \
    psycopg2-binary==2.9.9 \
    onnxruntime==1.18.0 \
    numpy==1.26.4 \
    boto3==1.34.120 \
    mlflow==2.13.0

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
