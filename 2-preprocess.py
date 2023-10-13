"""This module is responsible for preprocessing the data in the `data/raw` folder and storing it in the `data/processed` folder."""
import pandas as pd
from src.preprocessing import preprocess_fns

if __name__ == "__main__":
    res = pd.DataFrame(columns=["author", "date_published", "title", "text"])
    authors: set[str] = set()
    for fn in preprocess_fns:
        df, authors = fn()
        res = pd.concat([res, df])
        authors.update(authors)
    authors = pd.DataFrame({"author": sorted(list(authors))})
    authors.index.name = "class"
    authors.to_csv(f"data/processed/authors.csv")
    res["author"] = res["author"].apply(
        lambda x: authors[authors["author"] == x].index[0]
    )
    res.to_csv(f"data/processed/data.csv", index=False)
