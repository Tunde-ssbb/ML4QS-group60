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
    columns_to_drop = ['dist', 'pace']
    data = pd.read_csv(path)
    data = data.drop(columns=columns_to_drop)
    data['time'] = pd.to_datetime(data['time'])
    data['time'	] = data['time'].astype('int64')
    data['time'] = data['time'] - data['time'][0] 
    data['time'	] = data.index *0.1
    data= data.fillna(0)
    return data     

def accuracy(y_pred, y_true):
     if len(y_pred) != len(y_true):
          raise ValueError("pred and true must be same length")
     

     return np.sum(y_pred == y_true)/len(y_pred)

def get_chunk_sizes(min_size = 300, max_size = 3000):
    return np.arange(min_size, max_size + 1, 100)

def cut_into_chunks(df, chunk_sizes):
    chunks = []
    start_index = 0
    while start_index < len(df):
        chunk_size = np.random.choice(chunk_sizes)
        end_index = min(start_index + chunk_size, len(df))
        if len(df) - end_index <300:
            end_index = len(df)
        chunk = df.iloc[start_index:end_index]
        if(len(chunk) > 0):
            chunks.append(chunk)
        # print(f'len chunk {len(chunk)} ')
        start_index = end_index
    return chunks

def hussel(data, min_chunk_size = 300, max_chunk_size = 3000):
    chunk_sizes = get_chunk_sizes(min_chunk_size, max_chunk_size)
    all_chunks = []

    for session_id, group in data.groupby('session_id'):
        chunks = cut_into_chunks(group, chunk_sizes)
        all_chunks.extend(chunks)

    # Shuffle all chunks
    # print(len(all_chunks))
    np.random.shuffle(all_chunks)

    # Concatenate the shuffled chunks into a single DataFrame
    return all_chunks

def shuffle_test_train(chunks, duration=7000):
    test_chunks = []
    train_chunks = []
    cumulative_test_duration = 0
    exp_lvls_in_test = set()
    test_duration = duration

    for chunk in chunks:
        if cumulative_test_duration < test_duration or len(exp_lvls_in_test) < 4:
            if chunk['exp_lvl'].min() not in exp_lvls_in_test or len(exp_lvls_in_test) == 4:
                test_chunks.append(chunk)
                cumulative_test_duration += len(chunk)
                exp_lvls_in_test.add(chunk['exp_lvl'].min())
            else:
                train_chunks.append(chunk)
        else:
            train_chunks.append(chunk)

    # print(len(train_chunks))
    # print(len(test_chunks))
    np.random.shuffle(train_chunks)
    return test_chunks, train_chunks

def get_validation(train_chunks):
    valid_chunks = []
    new_training = []
    total_duration = 0
    test_duration = 5000
    for chunk in train_chunks:
        if total_duration < test_duration:
            valid_chunks.append(chunk)
            total_duration += len(chunk)
        else:
            new_training.append(chunk)
    return valid_chunks, new_training



# Shuffle and replace chunks
data = read_and_preprocess("./src/machine_learning/fouried_data.csv")
outputpath = "./src/machine_learning/shuffled.csv"
# outputpath2 = "./src/machine_learning/shuffledhow.csv"
# all_chunks = hussel(data, 300, 3000)
# test_chunks, train_chunks = shuffle_test_train(all_chunks)

# # Concatenate the test and training chunks into separate DataFrames
# test_df = pd.concat(test_chunks).reset_index(drop=True)
# train_df = pd.concat(train_chunks).reset_index(drop=True)

# # Adjust time columns to maintain continuity
# test_df['time'] = np.linspace(0, (len(test_df) - 1) * 0.1, len(test_df))  # Assuming time step is 0.1 seconds
# train_df['time'] = np.linspace(0, (len(train_df) - 1) * 0.1, len(train_df)) 

# shuffled_df = pd.concat(all_chunks).reset_index(drop=True)
# shuffled_df['time'] = np.linspace(0, (len(shuffled_df) - 1) * 0.1, len(shuffled_df))

# print(len(test_df))
# print(len(train_df))

# shuffled_df.to_csv(outputpath)
# shuffled_df['session_id'].to_csv(outputpath2)



