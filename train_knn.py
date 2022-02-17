import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from data_preprocessing import pipeline

np.random.seed(42)

data = pd.read_csv("train.csv")
X_train = pd.DataFrame(pipeline.fit_transform(data), index=data.PassengerId)
y_train = data.Survived

parameters = {
    "n_neighbors": range(1, 10)
}

clf = GridSearchCV(KNeighborsClassifier(), param_grid=parameters)
clf.fit(X_train, y_train)
model = clf.best_estimator_
print(clf.best_params_)
print(model.score(X_train, y_train))

test_data = pd.read_csv("test.csv")
X_test = pd.DataFrame(pipeline.transform(test_data))
predictions = pd.DataFrame(model.predict(X_test), columns=["Survived"], index=test_data.PassengerId)
predictions.to_csv("knn_predictions.csv")