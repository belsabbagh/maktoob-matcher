"""This module is responsible for preprocessing the data in the `data/raw` folder and storing it in the `data/processed` folder."""
import pandas as pd
from src.preprocessing import preprocess_fns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif



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
    features = SelectKBest(metric, k=top_number).fit(df, df.index).get_support(indices=True)
    df = df.iloc[:, features]
    return df


def preprocess_all(vectorizer):
    res = pd.DataFrame(columns=["author", "date_published", "title", "text"])
    authors: set[str] = set()
    for fn in preprocess_fns:
        df, authors = fn()
        res = pd.concat([res, df])
        authors.update(authors)
    authors = make_authors_df(authors)
    res["author"] = res["author"].apply(lambda x: get_author_id(x, authors))
    text_vec = top_features(text2vec(res["text"], vectorizer, col_prefix="text"))
    title_vec = top_features(text2vec(res["title"], vectorizer, col_prefix="title"))
    res = pd.concat([res[["author", "date_published"]], text_vec, title_vec], axis=1)
    return res, authors


if __name__ == "__main__":
    vectorizers = [TfidfVectorizer(), CountVectorizer()]
    for vectorizer in vectorizers:
        vec_name = vectorizer.__class__.__name__
        df, authors = preprocess_all(vectorizer)
        authors.to_csv(f"data/processed/authors.csv")
        df.to_csv(f"data/processed/data_{vec_name}.csv", index=False)
