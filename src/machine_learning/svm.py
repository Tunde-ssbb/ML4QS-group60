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
from sklearn.model_selection import GridSearchCV
import time


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

data = pd.read_csv("./src/machine_learning/fouried_data.csv", index_col=0)

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

session_id = 2
X_train, y_train, X_test, y_test = train_test_split_full_session(X, y, session_id)

print(X_test)
print(y_test)

print(f"test set session id: {session_id}")
print(f"exp lvl: {y_test.mean()}")


start = time.time()


# Set up the SVC classifier
svc = LinearSVC(verbose=0, C=100, max_iter=500, dual=0)
rfc = RandomForestClassifier(n_estimators=100, max_depth=10, verbose = 1)

model = svc 


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

# Setting up grid search
print(pipeline.get_params().keys())
param_grid = {'svc__estimator__C':[10,100,1000],'svc__estimator__max_iter':[500, 1000]}
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')


# Fit the pipeline to the training data
grid.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % grid.best_score_)
print(grid.best_params_)

# Make predictions on the test data
y_pred = grid.predict(X_test)

# Evaluate the model
for i in range(5):
    print(f"count {i}: {(y_pred == i).sum()}")

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# If you want to see which features were selected:
selected_features_kbest = X_train.columns.values[pipeline.named_steps.base.steps[1][1].get_support()]
print('Selected features by SelectKBest:', selected_features_kbest)
selected_features_sfs = selected_features_kbest[pipeline.named_steps.base.steps[2][1].get_support()]
print('Selected features by SequentialFeatureSelector:', selected_features_sfs)

end = time.time()

print(f"time: {end-start}s")