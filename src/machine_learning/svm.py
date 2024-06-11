import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# define a custum train test split to preserve temporality of data
def train_test_split(data, fraction):
    sessions = data['session_id'].unique()

    test = []
    train = []

    # per session:
    for session_id in sessions:


        # extract session data only
        session_data = data.loc[data['session_id'] == session_id]
        n_train_points = int(len(session_data['time']) - round(len(session_data['time'])*fraction))

        #print(f"for session {session_id} of len {len(session_data)}, the number of training points is {n_train_points}")

        session_train = session_data[:n_train_points]
        session_test = session_data[n_train_points:]

        test.append(session_test)
        train.append(session_train)
    
    train_data = pd.concat(train, axis = 0)
    test_data = pd.concat(test, axis = 0)

    return train_data, test_data

data = pd.read_csv("./fouried_data.csv")


# Assuming 'data' is your DataFrame and 'exp_lvl' is the target column
# Make sure the 'time' column is in datetime format
data['time'] = pd.to_datetime(data['time'])

# Preprocess the time column to extract features
data['time'] = data['time'].astype('int64')

data = data.fillna(0)



train, test = train_test_split(data, 0.2)

print(f" train size {train.shape}; test size {test.shape}")

# Assuming 'data' is your DataFrame and 'exp_lvl' is the target column
X_train = train.drop(columns='exp_lvl')  # Features
y_train = train['exp_lvl']  # Target

X_test = test.drop(columns='exp_lvl')  # Features
y_test = test['exp_lvl']  # Target


print(data.iloc[:, [236, 290]])

# Set up the SVC classifier
svc = SVC(kernel='linear')

# Set up the pipeline with StandardScaler and SequentialFeatureSelector
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('select_k_best', SelectKBest(f_classif, k=50)),  # Select top 10 features
    ('sfs', SequentialFeatureSelector(svc, n_features_to_select=25, direction='forward')),
    ('svc', svc)
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
