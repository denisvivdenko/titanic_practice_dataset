import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV

from data_preprocessing import pipeline

np.random.seed(42)

data = pd.read_csv("train.csv")
X_train = pd.DataFrame(pipeline.fit_transform(data), index=data.PassengerId)
y_train = data.Survived

dmatrix = xgb.DMatrix(data=X_train, label=y_train)
gbm_param_grid = {
    "learning_rate": [0.01, 0.1, 0.5, 0.9],
    "n_estimators": range(50, 250, 50),
    "subsample": [0.3, 0.5, 0.9],
    "objective": ['binary:logistic']
}

clf = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=gbm_param_grid, scoring='roc_auc', cv=5, verbose=10)
clf.fit(X_train, y_train)
model = clf.best_estimator_
print(model.score(X_train, y_train))
print(clf.best_score_)
print(clf.best_params_)

test_data = pd.read_csv("test.csv")
X_test = pd.DataFrame(pipeline.transform(test_data))
predictions = pd.DataFrame(model.predict(X_test), columns=["Survived"], index=test_data.PassengerId)
predictions.to_csv("xgboost_predictions.csv")