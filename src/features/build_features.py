"""
TF-IDF özellik çıkarımı için vektörleştiriciyi eğitir ve metinleri dönüştürür.
"""

import yaml
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Azerbaycanca stopwords listesi (sabit)
AZERBAIJANI_STOPWORDS = [
    "və", "ya", "ya da", "ilə", "üçün", "kimi", "tək", "qədər",
    "amma", "lakin", "ancaq", "çünki", "belə ki",
    "bu", "o", "şu", "bunlar", "onlar", "belə", "elə",
    "bir", "hər", "bütün", "heç", "çox", "az", "daha", "ən",
    "da", "də", "ki", "isə",
    "olan", "olur", "olar", "idi", "idir", "imiş", "olaraq",
    "mən", "sən", "o", "biz", "siz", "onlar",
    "məni", "səni", "onu", "bizi", "sizi", "onları",
    "indi", "sonra", "əvvəl", "həmişə", "tez-tez", "bəzən", "tez",
    "nə", "niyə", "necə", "harada", "kim", "hansı",
    "var", "yox", "beləliklə", "demək", "çünki"
]

def load_config(config_path: str = "config/model_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_tfidf_vectorizer(config: dict):
    """YAML konfigürasyonuna göre TF-IDF vektörleştirici oluşturur."""
    tfidf_params = config["feature_engineering"]["vectorizer"]["params"]

    if "ngram_range" in tfidf_params:
        ngram_str = tfidf_params["ngram_range"]
        if isinstance(ngram_str, str):
            ngram_tuple = tuple(map(int, ngram_str.split(",")))
            tfidf_params["ngram_range"] = ngram_tuple
    
    # Stopwords listesini parametrelere ekle
    tfidf_params["stop_words"] = AZERBAIJANI_STOPWORDS
    vectorizer = TfidfVectorizer(**tfidf_params)
    return vectorizer

def fit_and_save_vectorizer(train_texts, output_path: str, config_path: str = "config/model_config.yaml"):
    """
    Verilen metinler üzerinde TF-IDF vektörleştiriciyi eğitir ve diske kaydeder.
    """
    config = load_config(config_path)
    vectorizer = create_tfidf_vectorizer(config)
    vectorizer.fit(train_texts)
    joblib.dump(vectorizer, output_path)
    print(f"Vectorizer saved to {output_path}")
    return vectorizer

def transform_texts(vectorizer, texts):
    """Eğitilmiş vektörleştirici ile metinleri TF-IDF matrisine dönüştürür."""
    return vectorizer.transform(texts)

def main():
    """
    Eğitim verisini okur, vektörleştiriciyi eğitir ve kaydeder.
    Not: Bu fonksiyon doğrudan DVC pipeline tarafından çağrılmaz;
          `train_model.py` içinde kullanılır. Ancak bağımsız test için bırakılmıştır.
    """
    train_df = pd.read_csv("data/processed/train_cleaned.csv")
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    fit_and_save_vectorizer(train_df["content"], vectorizer_path)

if __name__ == "__main__":
    main()
