from os import listdir, path
import pandas as pd


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


def preprocess():
    df = pd.DataFrame(columns=["text", "author"])
    for metadata, text in dataset_iter():
        record = pd.DataFrame({"text": [text], "author": [metadata["author_name"]]})
        df = pd.concat([df, record], ignore_index=True)
    return df
