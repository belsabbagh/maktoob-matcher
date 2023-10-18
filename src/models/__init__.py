import tensorflow as tf


# TODO: Add all model builders here


def nn():
    pass


def cnn():
    pass


def lstm():
    pass


def nb():
    pass


def dt():
    pass


def svm():
    pass


all_model_builders = {
    "NeuralNetwork": nn,
    "ConvolutionalNeuralNetwork": cnn,
    "LongShort-TermMemory": lstm,
    "NaiveBayes": nb,
    "DecisionTree": dt,
    "SupportVector Machine": svm,
}
