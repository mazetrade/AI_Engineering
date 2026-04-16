FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["python", "train.py"]
