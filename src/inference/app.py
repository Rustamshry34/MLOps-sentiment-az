"""
FastAPI servisi: Azerbaycanca duygu analizi için model inference.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import numpy as np
import os
from src.inference.model_loader import load_model_and_vectorizer_from_registry
from src.preprocessing.text_cleaning import clean_text  # Aynı preprocessing!

# --------------------------------------------------
# MLflow configuration (KRİTİK)
# --------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI environment variable is not set")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri())

# Model ve vectorizer global olarak yüklenir (sunucu başlatıldığında bir kez)
try:
    model, vectorizer = load_model_and_vectorizer_from_registry(alias="production")
    app = FastAPI(title="Azerbaijani Sentiment Analysis API", version="1.0")
except Exception as e:
    raise RuntimeError(f"Failed to initialize model from MLflow: {e}")

class PredictionRequest(BaseModel):
    texts: Union[str, List[str]]

class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]  # Her sınıf için olasılık (MultinomialNB.predict_proba)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Girdiyi her zaman liste haline getir
        if isinstance(request.texts, str):
            texts = [request.texts]
        else:
            texts = request.texts

        if not texts:
            raise HTTPException(status_code=400, detail="Input text list is empty.")

        # Metinleri temizle — TAM OLARAK EĞİTİMDEKİ GİBİ!
        cleaned_texts = [clean_text(text) for text in texts]

        # Boş metin kontrolü
        if any(len(t.strip()) == 0 for t in cleaned_texts):
            # Opsiyonel: boş metinleri varsayılan tahminle doldurabilirsiniz,
            # ama şimdilik hataya çevirelim.
            raise HTTPException(status_code=400, detail="One or more inputs became empty after cleaning.")

        # Vektörleştirme ve tahmin
        X = vectorizer.transform(cleaned_texts)
        predictions = model.predict(X).tolist()
        probabilities = model.predict_proba(X).tolist()

        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}
