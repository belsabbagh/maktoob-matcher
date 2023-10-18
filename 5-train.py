"""This module is responsible for training the model and storing the trained model in the `out/models` folder."""
import pandas as pd
from src.models import all_model_builders


def save_model(model, name):
    path = f"out/models/{name}.keras"
    model.save(path)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/data_TfidfVectorizer.csv")
    X, y = df.drop(columns=["author"], inplace=False), df["author"]
    for name, model_builder in all_model_builders:
        model = model_builder()
        model.fit(X, y)
        save_model(model, name)
