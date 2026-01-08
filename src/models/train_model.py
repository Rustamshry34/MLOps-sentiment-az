"""
Model eğitim ve değerlendirme pipeline'ı.
MLflow entegrasyonu ile deney takibi ve model kayıt defteri desteği içerir.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import yaml

# Modüllerimizi import edelim
from src.features.build_features import fit_and_save_vectorizer, transform_texts
from src.features.build_features import load_config as load_feature_config

def load_training_config(config_path: str = "config/model_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # --- Konfigürasyonları yükle ---
    config = load_training_config()
    feature_config = config["feature_engineering"]
    train_config = config["training"]

    # --- MLflow ayarları ---
    mlflow.set_tracking_uri("file://./mlruns")  # GitHub Actions'ta lokal tracking
    mlflow.set_experiment("az_sentiment_nb")

    with mlflow.start_run():
        # --- Veriyi yükle ---
        train_df = pd.read_csv("data/processed/train_cleaned.csv")
        X_raw = train_df["content"]
        y = train_df["score"]

        # --- TF-IDF vektörleştiriciyi eğit ve kaydet ---
        vectorizer_path = "models/tfidf_vectorizer.pkl"
        vectorizer = fit_and_save_vectorizer(X_raw, vectorizer_path, "config/model_config.yaml")

        # --- Özellikleri dönüştür ---
        X_tfidf = transform_texts(vectorizer, X_raw)

        # --- Eğitim/test bölünmesi ---
        test_size = train_config["split"]["test_size"]
        random_state = train_config["split"]["random_state"]
        stratify = train_config["split"]["stratify"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )

        # --- Modeli oluştur ve eğit ---
        model_params = train_config["model"]["params"]
        model = MultinomialNB(**model_params)
        model.fit(X_train, y_train)

        # --- Tahminler ---
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # --- Metrikleri hesapla ---
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred)

        # Cross-validation
        cv_folds = train_config["evaluation"]["cv_folds"]
        cv_scores = cross_val_score(model, X_tfidf, y, cv=cv_folds, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        # --- Metrikleri topla ---
        metrics = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "cv_mean_accuracy": cv_mean,
            "cv_std_accuracy": cv_std,
        }

        # --- MLflow'a logla ---
        # Parametreler
        mlflow.log_params({
            "vectorizer__ngram_range": tuple(feature_config["vectorizer"]["params"]["ngram_range"]),
            "vectorizer__max_features": feature_config["vectorizer"]["params"]["max_features"],
            "vectorizer__min_df": feature_config["vectorizer"]["params"]["min_df"],
            "vectorizer__max_df": feature_config["vectorizer"]["params"]["max_df"],
            "model_type": train_config["model"]["type"],
            "test_size": test_size,
            "random_state": random_state,
            "cv_folds": cv_folds,
        })

        # Metrikler
        mlflow.log_metrics(metrics)

        # Sınıflandırma raporu (opsiyonel: artifact olarak)
        report = classification_report(y_test, y_pred, output_dict=True)
        with open("metrics/classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("metrics/classification_report.json")

        # --- Model ve vectorizer'ı MLflow'a kaydet ---
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(vectorizer_path, "vectorizer")

        # --- DVC için metrics.json ---
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        with open(metrics_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # --- Modeli diske kaydet (standalone kullanım için) ---
        model_path = "models/sklearn_model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        print("\n=== Training Summary ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
