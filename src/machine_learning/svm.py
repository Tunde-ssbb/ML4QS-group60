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

from src.machine_learning.util import train_test_split_full_session, read_and_preprocess
	
data = read_and_preprocess()

target = 'exp_lvl'

X = data.drop(columns=target)
y = data[target]

session_id = 23
"""
    session id 1 -> 1000, 1000 with 100% accuracy
"""
X_train, y_train, X_test, y_test = train_test_split_full_session(X, y, session_id)

print(X_test)
print(y_test)




start = time.time()


# Set up the SVC classifier
svc = LinearSVC(verbose=0, C=100, max_iter=1000, dual=False)
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
#param_grid = {'base__sfs__estimator__C': [10, 100, 1000], 'base__sfs__estimator__max_iter': [500, 1000]}
#grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1,scoring='accuracy')


# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)
#print("Best parameter (CV score=%0.3f):" % grid.best_score_)
#print(grid.best_params_)
#print(f'Total results: {grid.cv_results_}')

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

print(f"test set session id: {session_id}")
print(f"exp lvl: {y_test.mean()}")

# Evaluate the model
for i in range(5):
    print(f"count {i}: {(y_pred == i).sum()}")

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

#print(grid)
print(pipeline)

# If you want to see which features were selected:
selected_features_kbest = X_train.columns.values[pipeline.named_steps.base.steps[1][1].get_support()]
print('Selected features by SelectKBest:', selected_features_kbest)
selected_features_sfs = selected_features_kbest[pipeline.named_steps.base.steps[2][1].get_support()]
print('Selected features by SequentialFeatureSelector:', selected_features_sfs)

end = time.time()

print(f"time: {end-start}s")