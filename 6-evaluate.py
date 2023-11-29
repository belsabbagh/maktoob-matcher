"""This module is responsible for evaluating the model and storing the results in the `out/eval` folder."""
import os
from timeit import default_timer
from matplotlib import pyplot as plt
import numpy as np
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


def train_test_iter(X, y, kf):
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, y_train, X_test, y_test


def cross_validate(model, X, y, k=5, labels=None, is_nn=False):
    kf = KFold(n_splits=k)
    scores = {"label": [], "precision": [], "recall": [], "f1": []}
    cmats = []
    for X_train, y_train, X_test, y_test in train_test_iter(X, y, kf):
        # print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        y_hat = np.argmax(y_hat, axis=1) if is_nn else y_hat
        cmat = mt.confusion_matrix(y_test, y_hat, labels=labels)
        cmats.append(cmat)
        precision = cmat.diagonal() / cmat.sum(axis=0)
        recall = cmat.diagonal() / cmat.sum(axis=1)
        f1 = 2 * precision * recall / (precision + recall)
        for i, s in enumerate(list(zip(precision, recall, f1))):
            scores['label'].append(i)
            scores['precision'].append(s[0])
            scores['recall'].append(s[1])
            scores['f1'].append(s[2])
    cmats = np.add.reduce(cmats)
    scores = pd.DataFrame.from_dict(scores).groupby("label").mean().reset_index()
    scores.index = scores["label"]
    scores.drop(columns=["label"], inplace=True)
    return scores, cmats


def score(y, y_hat, labels):
    scores = mt.classification_report(y, y_hat, labels=labels, output_dict=True)
    scores["accuracy"] = mt.accuracy_score(y, y_hat)
    return scores


def files_iter():
    for f in os.listdir("data/processed/aninis"):
        if not f.startswith("data"):
            continue
        yield f


if __name__ == "__main__":
    count = 0
    start = default_timer()
    for filename in files_iter():
        vec, selection_method = filename.removesuffix(".csv").split("_")[1:]
        print(f"Training on vectorizer {vec} and selection method {selection_method}")
        df = pd.read_csv(f"data/processed/aninis/{filename}")
        df["date_published"] = pd.to_datetime(df["date_published"]).apply(
            datetime_to_float
        )
        X, y = (
            df.drop(columns=["author", "date_published"], inplace=False),
            df["author"],
        )
        nfeatures = X.shape[1]
        print("Number of features:", nfeatures)
        for name, model_builder in all_model_builders.items():
            print(
                f"Training {name} on vectorizer {vec} and selection method {selection_method}..."
            )
            savename = f"{name}_{vec}_{selection_method}"
            model = model_builder(nfeatures, 10)
            if name in each_model_parameter_grid.keys():
                grid_search = HalvingGridSearchCV(
                    model,
                    each_model_parameter_grid[name],
                    scoring="f1_macro",
                    n_jobs=-1,
                    verbose=1,
                )
                grid_search.fit(X, y)
                print(f'Best parameters For {name}:', grid_search.best_params_ )
                model = grid_search.best_estimator_
            scores, cmat = cross_validate(
                model,
                X,
                y,
                labels=df["author"].unique(),
                is_nn=name
                in [
                    "NeuralNetwork",
                    "ConvolutionalNeuralNetwork",
                    "LongShortTermMemory",
                ],
            )
            cmat = mt.ConfusionMatrixDisplay(cmat).plot(text_kw={"size": 3})
            plt.savefig(f"out/plots/confusion-matrix/{savename}.png", dpi=300)
            scores.to_csv(f"out/eval/cross-validation/{savename}.csv")
            for metric in ["precision", "recall", "f1"]:
                folder = f"out/plots/cross-validation"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                scores.plot(y=[metric])
                plt.savefig(f"{folder}/{metric}_{savename}.png", dpi=300)
        count += 1
    print(
        f"Fitted and cross-validated 6 models on {count} datasets in {default_timer()-start} seconds."
    )
