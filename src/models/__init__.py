import tensorflow as tf
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


N = 100
DEFAULT_INPUT = (N, 1)
DEFAULT_OUTPUT = 10


def nn(input_shape=DEFAULT_INPUT, output_shape=DEFAULT_OUTPUT):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(output_shape, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def cnn(input_shape=DEFAULT_INPUT, output_shape=DEFAULT_OUTPUT):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Reshape((input_shape, 1)),
            tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(output_shape, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def lstm(input_shape=DEFAULT_INPUT, output_shape=DEFAULT_OUTPUT):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Reshape((input_shape, 1)),
            # tf.keras.layers.Embedding(input_dim=11677, output_dim=64),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(output_shape, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def nb(*args):
    return GaussianNB()


def dt(*args):
    return DecisionTreeClassifier()


def svm(*args):
    return SVC()


all_model_builders = {
    "GaussianNB": nb,
    "DecisionTreeClassifier": dt,
    "SVC": svm,
    "NeuralNetwork": nn,
    "ConvolutionalNeuralNetwork": cnn,
    "LongShortTermMemory": lstm,
}

each_model_parameter_grid = {
    "GaussianNB": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        "priors": [None, [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]],
    },
    "DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 3, 4, 5],
    },
    "SVC": {
        "C": [0.1, 1, 10, 100, 1000],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
        "shrinking": [True, False],
        "decision_function_shape": ["ovo", "ovr"],
    },
}
