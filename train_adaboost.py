from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

from data_preprocessing import pipeline

np.random.seed(42)

data = pd.read_csv("train.csv")
X_train = pd.DataFrame(pipeline.fit_transform(data), index=data.PassengerId)
y_train = data.Survived

abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

parameters = {'base_estimator__max_depth':[i for i in range(2, 11, 2)],
              'base_estimator__min_samples_leaf':[5, 10],
              'n_estimators':[10, 50, 250, 300],
              'learning_rate':[0.01, 0.1]}

clf = GridSearchCV(abc, parameters, verbose=10, scoring = 'roc_auc')
clf.fit(X_train, y_train)
model = clf.best_estimator_

print(model.score(X_train, y_train))

test_data = pd.read_csv("test.csv")
X_test = pd.DataFrame(pipeline.transform(test_data))
predictions = pd.DataFrame(model.predict(X_test), columns=["Survived"], index=test_data.PassengerId)
predictions.to_csv("adaboost_predictions.csv")