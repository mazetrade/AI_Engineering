#Imports 

import mlflow
import mlflow.pytorch
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer
import sqlite3
import os
from datetime import datetime

#Define the data models 

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    timestamp: str
    
#Initialize the app and load the model

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts sentiment of text using DistilBERT",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer

    print("Loading model from MLflow...")
    mlflow.set_tracking_uri("http://mlflow:5000")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("sentiment-analysis")

    if experiment is None:
        raise Exception("No experiment found. Run training first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if not runs:
        raise Exception("No runs found. Run training first.")

    best_run_id = runs[0].info.run_id
    print(f"Loading model from run: {best_run_id}")

    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    tokenizer_path = mlflow.artifacts.download_artifacts(
        run_id=best_run_id,
        artifact_path="tokenizer"
    )
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

    print("Model loaded successfully!")
    
#Database setup for monitoring 

def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(text: str, label: str, confidence: float, timestamp: str):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (text, label, confidence, timestamp) VALUES (?, ?, ?, ?)",
        (text, label, confidence, timestamp)
    )
    conn.commit()
    conn.close()
    
#Startup event 

@app.on_event("startup")
async def startup_event():
    init_db()
    load_model()
    
#Prediction endpoint 

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Tokenize the input text
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        label = "positive" if predicted_class == 1 else "negative"
        timestamp = datetime.now().isoformat()

        # Log to database for monitoring
        log_prediction(request.text, label, confidence, timestamp)

        return PredictionResponse(
            text=request.text,
            label=label,
            confidence=round(confidence, 4),
            timestamp=timestamp
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#Health check endpoint 

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API is running"}

