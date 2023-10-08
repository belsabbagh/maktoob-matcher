"""This module is responsible for preprocessing the data in the `data/raw` folder and storing it in the `data/processed` folder."""
import pandas as pd
from src.preprocessing import preprocess_fns

if __name__ == "__main__":
    res = pd.DataFrame(columns=["text", "author"])
    for fn in preprocess_fns:
        res = pd.concat([res, fn()])
    res.to_csv(f"data/processed/data.csv", index=False)
