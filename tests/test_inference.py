"""
Inference test: Model ve vectorizer lokal dosyalardan yüklenir.
Bu test, DVC pipeline'ı çalıştıktan sonra (dvc repro) geçerli olur.
"""

import joblib
from pathlib import Path
import sys
import numpy as np


# src'yi PYTHONPATH'a ekle (test bağımsız çalışmalı)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.text_cleaning import clean_text


def test_inference_loads_and_predicts():
    # Model ve vectorizer yolları
    model_path = Path("models/sklearn_model.pkl")
    vectorizer_path = Path("models/tfidf_vectorizer.pkl")

    assert model_path.exists(), f"Model not found at {model_path}"
    assert vectorizer_path.exists(), f"Vectorizer not found at {vectorizer_path}"

    # Yüklemeler
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Test metni (Azerbaycanca örnek)
    raw_text = "Məhsul çox yaxşıdır! 😊"
    cleaned = clean_text(raw_text)
    assert cleaned == "məhsul çox yaxşıdır!"

    # Vektörleştirme ve tahmin
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)
    proba = model.predict_proba(X)

    # Tahmin mantıklı mı?
    assert pred.shape == (1,)
    assert proba.shape[0] == 1
    assert np.isclose(proba.sum(), 1.0, atol=1e-8)

    print(f"✅ Test passed: '{raw_text}' → prediction={pred[0]}, proba={proba[0].round(3)}")


if __name__ == "__main__":
    test_inference_loads_and_predicts()
