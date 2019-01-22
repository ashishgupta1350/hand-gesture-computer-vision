import pandas as pd
import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout, Flatten

# import autokeras as ak

x_train = pd.read_csv("images.csv", delimiter=",")
y_train = pd.read_csv("labels.csv", delimiter=",")

model = Sequential()

n_cols = x_train.shape[1]

# add model layers
model.add(Dense(100, activation='relu', input_dim= 2000))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))

model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])


model.fit(x_train, y=y_train, epochs=20, batch_size=20)
