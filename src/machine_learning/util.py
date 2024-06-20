import pandas as pd
import numpy as np

# define a custum train test split to preserve temporality of data
def train_test_split_session_fraction(data, X, y, fraction):
    sessions = data['session_id'].unique()

    X_test_data = []
    X_train_data = []
    y_train_data = []
    y_test_data = []

    # per session:
    for session_id in sessions:


        # extract session data only
        session_X = X.loc[X['session_id'] == session_id]
        session_y = y.loc[X['session_id'] == session_id]

        n_train_points = int(len(session_X['time']) - round(len(session_X['time']) * fraction))

        #print(f"for session {session_id} of len {len(session_data)}, the number of training points is {n_train_points}")

        session_X_train = session_X[:n_train_points]
        session_X_test = session_X[n_train_points:]

        session_y_train = session_y[:n_train_points]
        session_y_test = session_y[n_train_points:]

        X_test_data.append(session_X_test)
        y_test_data.append(session_y_test)

        X_train_data.append(session_X_train)
        y_train_data.append(session_y_train)
    
    X_train = pd.concat(X_train_data, axis = 0)
    y_train = pd.concat(y_train_data, axis = 0)

    X_test = pd.concat(X_test_data, axis = 0)
    y_test = pd.concat(y_test_data, axis = 0)

    return X_train, y_train, X_test, y_test

def train_test_split_full_session(X, y, test_session_id):
    X_test = X.loc[X['session_id'] == test_session_id]
    y_test = y.loc[X['session_id'] == test_session_id]

    X_train = X.loc[X['session_id'] != test_session_id]
    y_train = y.loc[X['session_id'] != test_session_id]

    return X_train, y_train, X_test, y_test

def read_and_preprocess(path) -> pd.DataFrame:
	columns_to_drop = ['dist' , 'pace']
	
	data = pd.read_csv(path)
	
	data = data.drop(columns = columns_to_drop)
	
	# Assuming 'data' is your DataFrame and 'exp_lvl' is the target column
	# Make sure the 'time' column is in datetime format
	data['time'] = pd.to_datetime(data['time'])
	
	# Preprocess the time column to extract features
	data['time'] = data['time'].astype('int64')
	
	data = data.fillna(0)
     
	return data


def accuracy(y_pred, y_true):
     if len(y_pred) != len(y_true):
          raise ValueError("pred and true must be same length")
     

     return np.sum(y_pred == y_true)/len(y_pred)