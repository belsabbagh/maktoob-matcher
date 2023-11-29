import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif as fclassif


def text2vec(text, vectorizer, vectorizer_kwargs=None, col_prefix=""):
    if vectorizer_kwargs is not None:
        vectorizer_kwargs = {}
    vec = vectorizer.fit_transform(text).toarray()
    words = vectorizer.get_feature_names_out()
    df = pd.DataFrame(vec, columns=[col_prefix + i for i in words])
    return df


def top_features(df, percentage=0.1, metric=chi2) -> pd.DataFrame:
    top_number = int(len(df.columns) * percentage)
    features = (
        SelectKBest(metric, k=top_number).fit(df, df.index).get_support(indices=True)
    )
    df = df.iloc[:, features]
    return df


if __name__ == "__main__":
    vectorizers = [TfidfVectorizer(), CountVectorizer()]
    feature_selection_methods = [chi2, fclassif]
    for vectorizer in vectorizers:
        vec_name = vectorizer.__class__.__name__
        for feature_selection_method in feature_selection_methods:
            class_index = 0
            df = pd.DataFrame(columns=["text", "author"])
            for author_file in os.listdir("data/raw"):
                with open("data/raw/" + author_file, "r") as f:
                    articles = json.load(f)
                text = [
                    article["title"] + " " + article["content"] for article in articles
                ]
                author = [class_index] * len(text)
                df = pd.concat([df, pd.DataFrame({"text": text, "author": author})])
                class_index += 1
            text_vec = text2vec(
                df["text"],
                vectorizer,
                vectorizer_kwargs={"ngram_range": (1, 1)},
                col_prefix=vec_name + "_",
            )
            text_vec = top_features(text_vec, metric=feature_selection_method)
            authors = df[["author"]].reset_index(drop=True)
            text_vec = text_vec.reset_index(drop=True)
            pd.concat([authors, text_vec], axis=1, ignore_index=True).to_csv(
                f"data/processed/{vec_name}_{feature_selection_method.__name__}.csv",
                index=False,
            )
