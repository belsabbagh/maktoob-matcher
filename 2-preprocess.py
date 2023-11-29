"""This module is responsible for preprocessing the data in the `data/raw` folder and storing it in the `data/processed` folder."""
import json
import pandas as pd
from src.preprocessing import preprocess_fns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif as fclassif


def make_authors_df(authors):
    df = pd.DataFrame({"author": sorted(list(authors))})
    df.index.name = "class"
    return df


def text2vec(text, vectorizer, vectorizer_kwargs=None, col_prefix=""):
    if vectorizer_kwargs is not None:
        vectorizer_kwargs = {}
    vec = vectorizer.fit_transform(text).toarray()
    words = vectorizer.get_feature_names_out()
    df = pd.DataFrame(vec, columns=[col_prefix + i for i in words])
    return df


def get_author_id(author, authors):
    return authors[authors["author"] == author].index[0]


def top_features(df, percentage=0.1, metric=chi2) -> pd.DataFrame:
    top_number = int(len(df.columns) * percentage)
    features = (
        SelectKBest(metric, k=top_number).fit(df, df.index).get_support(indices=True)
    )
    df = df.iloc[:, features]
    return df


def preprocess_all(
    vectorizer, feature_selection_method, feature_selection_percentage=0.1
):
    res = pd.DataFrame(columns=["author", "date_published", "text"])
    authors: set[str] = set()
    for fn in preprocess_fns:
        df, authors = fn()
        res = pd.concat([res, df])
        authors.update(authors)
    authors = make_authors_df(authors)
    res["author"] = res["author"].apply(lambda x: get_author_id(x, authors))
    merge_title = lambda x: x["title"] + " " + x["text"]
    res["text"] = res.apply(merge_title, axis=1)
    res.drop(columns=["title"], inplace=True)
    df = text2vec(res["text"], vectorizer)
    text_vec = top_features(
        df, percentage=feature_selection_percentage, metric=feature_selection_method
    )
    res = pd.concat([res[["author", "date_published"]], text_vec], axis=1)
    return res, authors


if __name__ == "__main__":
    vectorizers = [TfidfVectorizer(), CountVectorizer()]
    feature_selection_methods = [chi2, fclassif]
    for vectorizer in vectorizers:
        vec_name = vectorizer.__class__.__name__
        for feature_selection_method in feature_selection_methods:
            df, authors = preprocess_all(vectorizer, feature_selection_method)
            authors.to_csv(f"data/processed/aninis/authors.csv")
            df.to_csv(
                f"data/processed/aninis/data_{vec_name}_{feature_selection_method.__name__.replace('_', '-')}.csv",
                index=False,
            )
