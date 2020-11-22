import tensorflow as tf
import keras
from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import PReLU, ELU
from keras.initializers import Constant

graph = None

def build_model():
    num_classes = 2
    keras.backend.clear_session()
    
    model = Sequential()
    model.add(keras.layers.Dense(1024, input_shape=(1538,)))
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Dropout(0.7))
    model.add(keras.layers.Dense(512))
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Dropout(0.7))
    model.add(keras.layers.Dense(128))
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Dropout(0.3))
    model.add(keras.layers.Dense(64))
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Dropout(0.1))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))
    return model

def load_weight(model, path):
    global graph
    graph = tf.get_default_graph()
    model.load_weights(path)

def predict_verification(model, X):
    global graph
    with graph.as_default():
        return model.predict(X)