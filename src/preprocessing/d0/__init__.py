from os import listdir, path
import pandas as pd

from src.preprocessing.text import preprocess_text


def parse_properties(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    properties = {}
    for line in lines:
        if line.startswith("#"):
            continue
        key, value = line.split("=")
        key = key.strip()
        value = value.strip()
        properties[key] = value
    return properties


RAW_PATH = "data/raw/Authorship attribution data"


def dataset_iter():
    for folder in listdir(RAW_PATH):
        for i in range(0, 10):
            metadata = parse_properties(
                path.join(RAW_PATH, folder, f"{folder}_{i}.properties")
            )
            text = ""
            with open(
                path.join(RAW_PATH, folder, f"sample{i}.txt"),
                "r",
                encoding="utf-8",
            ) as f:
                text = f.read()
            text = text.replace("\n", " ")
            yield metadata, text


def extract_date(metadata):
    date = f"{metadata['article_year']}-{metadata['article_month']}-{metadata['article_day']}"
    date = pd.to_datetime(date, format="%Y-%m-%d")
    return date


def aninis_preprocess():
    df = pd.DataFrame(columns=["author", "date_published", "title", "text"])
    authors: set[str] = set()
    for meta, text in dataset_iter():
        record = pd.DataFrame(
            {
                "author": [meta["class_index"]],
                "date_published": [extract_date(meta)],
                "title": [preprocess_text(meta["article_title"])],
                "text": [preprocess_text(text)],
            }
        )
        authors.add(meta["author_name"])
        df = pd.concat([df, record], ignore_index=True)
    return df, authors
