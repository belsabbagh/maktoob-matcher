import pandas as pd
import os


def _preprocess_folder(folder):
    df = pd.DataFrame(columns=["text", "author"])
    authors = set()
    for author in os.listdir(folder):
        authors.add(author)
        for file in os.listdir(os.path.join(folder, author)):
            with open(os.path.join(folder, author, file), "r") as f:
                text = f.read()
            new_record = pd.DataFrame([[text, author]], columns=["text", "author"])
            df = pd.concat([df, new_record], ignore_index=True)
    return df, authors


def c50_preprocess():
    df, authors = _preprocess_folder("data/raw/c50/C50train")
    df_test, _ = _preprocess_folder("data/raw/c50/C50test")
    return pd.concat([df, df_test], ignore_index=True), authors
