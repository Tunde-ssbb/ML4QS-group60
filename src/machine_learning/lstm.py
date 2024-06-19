import keras
from keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time

from util import train_test_split_full_session, read_and_preprocess, train_test_split_session_fraction

data = read_and_preprocess()

target = 'exp_lvl'

X = data.drop(columns=target)
yd = data[['time', target]]
yd = yd[target].astype('category')
y = pd.get_dummies(yd)

session_id = 23

X_train, y_train, X_test, y_test = train_test_split_session_fraction(data, X, y, 0.2)

print(X_test)
print(y_test)

print(f"test set session id: {session_id}")
print(f"exp lvl: {y_test.mean()}")

X_train_rs = X_train.values.reshape(1, X_train.shape[0], X_train.shape[1])

# Reshape the test data to fit the LSTM input requirements
X_test_rs = X_test.values.reshape(1, X_test.shape[0], X_test.shape[1])

# Reshape y_train to match the number of samples
y_train_rs = y_train.values.reshape(1, y_train.shape[0], y_train.shape[1])
y_test_rs = y_test.values.reshape(1, y_test.shape[0], y_test.shape[1])
print(y_train_rs.shape)
print(y_test_rs.shape)

start = time.time()

model = keras.Sequential()
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy"
              , metrics=['acc']
              , optimizer="adam")

#model.summary()

model.fit(X_train_rs, y_train_rs, batch_size=1, epochs=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_rs, y_test_rs)
print(f'Test Loss: {loss:.2f}, Test Accuracy: {accuracy:.2f}')

# Predict on the test data
y_test_scores = model.predict(X_test_rs, verbose=1)
print(y_test_scores)
print(y_test_scores.shape)
y_pred = np.argmax(y_test_scores, axis=2)
print(y_pred)

#y_pred = np.argmax(y_test_scores, axis=1) #  np.where(y_test_prob > 0.5, 1, 0)

# Evaluate the model
for i in range(5):
    print(f"count {i}: {(y_pred == i).sum()}")

ta = pd.from_dummies(y_test).squeeze(axis = 1)

tr = y_pred.flatten()

print(f"actual labels ({len(ta)}): {ta}")
print(f"predicted labels ({len(tr)}): {tr}")


known_labels = set(ta.unique())
tr = np.array([label if label in known_labels else -1 for label in tr])

ta = tr[tr != -1]
tr = tr[tr != -1]

print(f"no. of actual labels after unknown labels removed: {len(ta)}")
print(f"no. of predicted labels after unknown labels removed: {len(tr)}")


accuracy = accuracy_score(ta, tr)
print(f'Accuracy: {accuracy:.2f}')

end = time.time()

print(f"time: {end-start}s")