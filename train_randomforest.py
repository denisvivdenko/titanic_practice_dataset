import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from data_preprocessing import pipeline

np.random.seed(42)

data = pd.read_csv("train.csv")
X_train = pd.DataFrame(pipeline.fit_transform(data), index=data.PassengerId)
y_train = data.Survived

parameters = {
    'bootstrap': [True, False],
    'max_depth': list(range(10, 100, 10)) + [None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': range(50, 250, 50)
}
classifier = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=5, verbose=10)
classifier.fit(X_train, y_train)

print("best_score: ", classifier.best_score_)
print("best_params: ", classifier.best_params_)
best_score:  0.8170673529596385
best_params = {'bootstrap': True, 'max_depth': 80, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}
model = RandomForestClassifier(**best_params)
model.fit(X_train, y_train)
print(model.score(X_train, y_train))

test_data = pd.read_csv("test.csv")
X_test = pd.DataFrame(pipeline.transform(test_data))
predictions = pd.DataFrame(model.predict(X_test), columns=["Survived"], index=test_data.PassengerId)
predictions.to_csv("randomforest_gridsearchcv_predictions.csv")