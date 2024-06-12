import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.multiclass import OneVsRestClassifier


# define a custum train test split to preserve temporality of data
def train_test_split_session_fraction(X, y, fraction):
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


columns_to_drop = ['dist' , 'pace']

data = pd.read_csv("./fouried_data.csv")

data = data.drop(columns = columns_to_drop)

# Assuming 'data' is your DataFrame and 'exp_lvl' is the target column
# Make sure the 'time' column is in datetime format
data['time'] = pd.to_datetime(data['time'])

# Preprocess the time column to extract features
data['time'] = data['time'].astype('int64')

data = data.fillna(0)

target = 'exp_lvl'

X = data.drop(columns=target)
y = data[target]

X_train, y_train, X_test, y_test = train_test_split_full_session(X, y, 0.2)

print(X_test)
print(y_test)


# Set up the SVC classifier
svc = LinearSVC(verbose=1, max_iter=1000, C=10)
rfc = RandomForestClassifier(n_estimators=100, max_depth=10, verbose = 1)

model = rfc


# Set up the pipeline with StandardScaler and SequentialFeatureSelector
base_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('select_k_best', SelectKBest(f_classif, k=100)),  # Select top 50 features
    ('sfs', SequentialFeatureSelector(model, n_features_to_select=25, direction='forward'))  # Select top 25 features sequentially
])

# Use OneVsRestClassifier to handle multiple binary classifications
pipeline = Pipeline([
    ('base', base_pipeline),
    ('svc', OneVsRestClassifier(model))
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# If you want to see which features were selected:
selected_features_kbest = X_train.columns[pipeline.named_steps['select_k_best'].get_support()]
selected_features_sfs = X_train.columns[pipeline.named_steps['sfs'].get_support()]
print('Selected features by SelectKBest:', selected_features_kbest)
print('Selected features by SequentialFeatureSelector:', selected_features_sfs)

