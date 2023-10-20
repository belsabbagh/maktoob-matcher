import tensorflow as tf


# TODO: Add all model builders here


N = None
input_shape = (N, 1)
output_shape = 10


def nn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    return model

def cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((input_shape[0], 1)),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    return model

def lstm():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Embedding(input_dim=input_shape, output_dim=64), 
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    return model


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
