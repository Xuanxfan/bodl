import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf

from matplotlib import pyplot as plt


def Cov_model():
    model.summary()
    model = Sequential()
    model.add(Conv1D(32, 2, padding='valid', input_shape=(10, 10), activation="relu"))
    model.add(Conv1D(62, 2, padding='valid', activation="relu"))
    model.add(Conv1D(128, 2, padding='valid', activation="relu"))
    model.add(Conv1D(256, 2, padding='valid', activation="relu"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    opt = Adam(lr=0.000001)
    model.compile(optimizer=opt, loss=tf.keras.losses.mean_absolute_error,
                  metrics=['accuracy'])
    # history = model.fit(x_train, y_train, epochs=500, validation_data=(x_val, y_val))

    return model

