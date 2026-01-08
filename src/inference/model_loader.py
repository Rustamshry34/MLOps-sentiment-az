"""
Model ve TF-IDF vektörleştiriciyi diskten yükler.
İleride MLflow model URI desteği eklenebilir.
"""

import joblib
from pathlib import Path
from sklearn.base import BaseEstimator

def load_model_and_vectorizer(
    model_path: str = "models/sklearn_model.pkl",
    vectorizer_path: str = "models/tfidf_vectorizer.pkl"
) -> tuple[BaseEstimator, object]:
    """
    Eğitilmiş modeli ve TF-IDF vektörleştiriciyi yükler.

    Returns:
        tuple: (model, vectorizer)
    """
    model_path = Path(model_path)
    vectorizer_path = Path(vectorizer_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer
