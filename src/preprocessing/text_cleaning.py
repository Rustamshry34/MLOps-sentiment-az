"""
Metin verisini temizler ve data/processed/ altına kaydeder.
"""

import pandas as pd
import regex
from pathlib import Path

def clean_text(text: str) -> str:
    """
    Azerbaycanca metin için özelleştirilmiş temizleme fonksiyonu.
    Emojiler, ! ve ? korunur; URL'ler, mention'lar ve gereksiz noktalama silinir.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    # URL'leri kaldır
    text = regex.sub(r"http\S+", " ", text)

    # Mention'ları kaldır (@kullanıcı)
    text = regex.sub(r"@\w+", " ", text)

    # Hashtag işaretini kaldır, kelimeyi bırak
    text = regex.sub(r"#", "", text)

    # Sayıları kaldır (tam sayılar)
    text = regex.sub(r"\b\d+\b", " ", text)

    # Sadece harfler, boşluk, ! ve ? kalsın (Unicode harfler dahil — \p{L})
    text = regex.sub(r"[^\p{L}\s!?]", " ", text)

    # Fazla boşlukları tek boşluğa indirge
    text = regex.sub(r"\s+", " ", text)

    return text.strip()

def main():
    input_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        input_path = input_dir / f"{split}.csv"
        output_path = output_dir / f"{split}_cleaned.csv"

        print(f"Processing {input_path}...")
        df = pd.read_csv(input_path)

        # NaN içerikleri düşür
        df = df.dropna(subset=["content"])

        # Temizleme uygula
        df["content"] = df["content"].apply(clean_text)

        # Boş metinleri filtrele (temizlemeden sonra oluşabilir)
        df = df[df["content"].str.len() > 0]

        df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    main()
