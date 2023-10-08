"""This module is responsible for preprocessing the data in the `data/raw` folder and storing it in the `data/processed` folder."""
from src.preprocessing import preprocess_fns

if __name__ == "__main__":
    for fn in preprocess_fns:
        df = fn()
    df.to_csv(f"data/processed/data.csv", index=False)
