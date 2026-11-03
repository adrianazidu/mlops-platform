FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.13.0 \
    scikit-learn==1.5.0 \
    numpy==1.26.4 \
    boto3==1.34.120

COPY train.py .

CMD ["python", "train.py"]
