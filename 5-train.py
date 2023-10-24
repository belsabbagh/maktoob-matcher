"""This module is responsible for training the model and storing the trained model in the `out/models` folder."""
import json
import os
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, KFold
from sklearn import metrics as mt
import tensorflow as tf
from src.models import all_model_builders
from src.models import each_model_parameter_grid
import datetime
import pickle
import warnings

warnings.filterwarnings("ignore")


def datetime_to_float(d):
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds = (d - epoch).total_seconds()
    return total_seconds


def save_model(model, name):
    path = f"out/models/{name}.keras"
    if name in ["SVC", "GaussianNB", "DecisionTreeClassifier"]:
        path = f"out/models/{name}.pkl"
        pickle.dump(model, open(path, "wb"))
        return
    model.save(path)


def cross_validate(model, X, y, k=10, labels=None, is_nn=False):
    kf = KFold(n_splits=k)
    scores = {
        "label": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        if is_nn:
            y_hat = tf.argmax(y_hat, axis=1)
        score_dict = score(y_test, y_hat, labels)
        for k, v in score_dict.items():
            scores[k].extend(v)
    scores = pd.DataFrame.from_dict(scores).groupby("label").mean().reset_index()
    scores.index = scores["label"]
    scores.drop(columns=["label"], inplace=True)
    return scores


def score(y, y_hat, labels):
    score_dict = {
        "label": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for label in labels:
        tmp_true = y == label
        tmp_pred = y_hat == label
        score_dict["label"].append(label)
        score_dict["accuracy"].append(mt.accuracy_score(tmp_true, tmp_pred))
        score_dict["precision"].append(mt.precision_score(tmp_true, tmp_pred))
        score_dict["recall"].append(mt.recall_score(tmp_true, tmp_pred))
        score_dict["f1"].append(mt.f1_score(tmp_true, tmp_pred))
    return score_dict


def files_iter():
    for f in os.listdir("data/processed"):
        if not f.startswith("data"):
            continue
        yield f


if __name__ == "__main__":
    for filename in files_iter():
        vec, selection_method = filename.removesuffix(".csv").split("_")[1:]
        print(f"Training on vectorizer {vec} and selection method {selection_method}")
        df = pd.read_csv(f"data/processed/{filename}")
        df["date_published"] = pd.to_datetime(df["date_published"]).apply(
            datetime_to_float
        )
        X, y = df.drop(columns=["author"], inplace=False), df["author"]
        for name, model_builder in all_model_builders.items():
            print(f"Training {name}...")
            model = model_builder(1168, 10)
            if name in each_model_parameter_grid.keys():
                grid_search = GridSearchCV(
                    model,
                    each_model_parameter_grid[name],
                    scoring="f1_macro",
                    n_jobs=-1,
                    verbose=1,
                )
                grid_search.fit(X, y)
                print(f'Best parameters For {name}:', grid_search.best_params_ )
                model = grid_search.best_estimator_
            scores = cross_validate(model, X, y, labels=df["author"].unique(), is_nn=name in ["NeuralNetwork", "ConvolutionalNeuralNetwork", "LongShortTermMemory"])
            scores.to_csv(f"out/eval/{name}_{vec}_{selection_method}.csv", index=False)
            model.fit(X, y)
            save_model(model, name)

