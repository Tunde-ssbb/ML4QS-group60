import keras
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
import numpy as np
import pandas as pd
import time

from src.machine_learning.util import train_test_split_full_session, read_and_preprocess

data = read_and_preprocess()

target = 'exp_lvl'

X = data.drop(columns=target)
y = data[target]

session_id = 23

X_train, y_train, X_test, y_test = train_test_split_full_session(data, X, y, session_id)

print(X_test)
print(y_test)

print(f"test set session id: {session_id}")
print(f"exp lvl: {y_test.mean()}")


start = time.time()

model = keras.Sequential()
model.add(LSTM(100, input_shape = (10, 28)))
model.add(Dropout(0.5))
model.add(Dense(5, activation="sigmoid"))
model.compile(loss="categorical_crossentropy"
              , metrics=['acc']
              , optimizer="adam")

model.summary()

model.fit(X_train, y_train)

model.evaluate(X_test, y_test)

y_test_prob = model.predict(X_test, verbose=1)

y_test_pred = np.where(y_test_prob > 0.5, 1, 0)

# Evaluate the model
for i in range(5):
    print(f"count {i}: {(y_pred == i).sum()}")

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')