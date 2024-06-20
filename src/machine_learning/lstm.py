import keras
from keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time
import util
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

def run_LSTM(data, session_id, random_seed = 42):
    np.random.seed(random_seed)

    target = 'exp_lvl' 

    X = data.drop(columns=target)
    yd = data[['time', target]]
    yd = yd[target].astype('category')
    y = pd.get_dummies(yd)

  
    X_train, y_train, X_test, y_test = util.train_test_split_full_session( X, y, session_id)

    scaler = StandardScaler()  # MinMaxScaler(feature_range=(-1,1))
    X_train = pd.DataFrame(scaler.fit_transform(X_train.values),
                                     index=X_train.index,
                                     columns=X_train.columns)
    # The Scaler is fit on the training set and then applied to the test set
    X_test = pd.DataFrame(scaler.transform(X_test.values),
                                    index=X_test.index,
                                    columns=X_test.columns)

    X_train = X_train.drop(columns = 'session_id')
    X_test = X_test.drop(columns = 'session_id')
    print(X_test)
    print(y_test)

    print(f"test set session id: {session_id}")
    print(f"exp lvl: {np.argmax(y_test.mean())}")

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
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.8))
    print(y.shape[1])
    model.add(Dense(y.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy"
                , metrics=['acc']
                , optimizer=Adam(learning_rate=0.1))

    #model.summary()

    model.fit(X_train_rs, y_train_rs, batch_size=16, epochs=20)

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

    acc =  util.accuracy(tr, ta)
    print(f"calculated accuracy: {acc}")

    end = time.time()

    print(f"time: {end-start}s")

    return acc, tr, np.argmax(y_test.mean())




best_25 = ['arm_gyr_z', 'arm_acc_x', 'leg_gyr_y', 'leg_acc_y', 'HR', 'session_id',
 'arm_gyr_y_freq_weighted', 'arm_gyr_z_max_freq', 'arm_acc_x_freq_weighted',
 'arm_acc_z_max_freq', 'arm_acc_z_pse', 'arm_acc_z_freq_0.0_Hz_ws_100',
 'arm_acc_z_freq_0.2_Hz_ws_100', 'leg_gyr_z_max_freq',
 'leg_acc_x_freq_0.0_Hz_ws_100', 'leg_acc_y_freq_weighted',
 'leg_acc_z_max_freq', 'arm_gyr_x_std', 'arm_acc_x_std',
 'arm_acc_z_std', 'leg_gyr_y_std', 'leg_gyr_z_std', 'arm_leg_max_diff',
 'acc_x_derivative_diff','arm_gyr_y_std']

base = ['exp_lvl', 'time']

data = util.read_and_preprocess("./src/machine_learning/fouried_data.csv")
print(data.columns)
for feature in data.columns:
    if not (feature in (best_25 + base)):
        data = data.drop(columns=feature)



print(data.shape)

session_ids = data['session_id'].unique()

print(f"Testing for sessions {session_ids}")
accuracies = []
trs = []
levels = []

for session_id in [31]:
    acc, tr, exp_lvl = run_LSTM(data, session_id)

    accuracies.append(acc)
    trs.append(np.mean(tr))
    levels.append(exp_lvl)

for session, accuracy, exp_lvl, tr in zip(session_ids, accuracies, levels, trs):
    print(session, accuracy, exp_lvl, tr, sep='\t\t')
    
#accuracy = accuracy_score(ta, tr)
#print(f'Accuracy: {accuracy:.2f}')

