"""
Veriyi Hugging Face dataset'inden indirir ve data/raw/ altına kaydeder.
"""

import pandas as pd
from pathlib import Path

def main():
    # Output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hugging Face raw CSV URLs
    base_url = "https://huggingface.co/datasets/hajili/azerbaijani_review_sentiment_classification/resolve/main/data"
    
    splits = {
        "train": f"{base_url}/train.csv",
        "test": f"{base_url}/test.csv"
    }

    for split_name, url in splits.items():
        print(f"Downloading {split_name} split from {url}...")
        df = pd.read_csv(url)
        output_path = output_dir / f"{split_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
