"""
MLflow Model Registry'den model ve vectorizer'ı yükler.
Vectorizer da MLflow artifact'i olarak kaydedildiği için oradan alınır.
"""

import mlflow
from pathlib import Path
import tempfile
import joblib
import os 

def load_model_and_vectorizer_from_registry(
    model_name: str = "az_sentiment_nb",
    alias: str = "production" 
):
    # MLflow tracking URI — production'da remote olabilir
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_uri)
    
    """
    MLflow Model Registry'den belirtilen stage'deki modeli ve vectorizer'ı yükler.
    """
    try:
        # Model URI'sini al
        model_uri = f"models:/{model_name}@{alias}"
        print(f"Loading model from URI: {model_uri}")

        # Modeli yükle
        model = mlflow.sklearn.load_model(model_uri)

        # Vectorizer artifact'ini yükle
        model_info = mlflow.models.get_model_info(model_uri)
        run_id = model_info.run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            # Vectorizer'ı artifact'ten indir
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="vectorizer/tfidf_vectorizer.pkl",
                dst_path=tmpdir
            )
            vectorizer_path = Path(tmpdir) / "tfidf_vectorizer.pkl"
            vectorizer = joblib.load(vectorizer_path)

        return model, vectorizer

    except Exception as e:
        raise RuntimeError(f"Failed to load model/vectorizer from MLflow registry: {e}")
