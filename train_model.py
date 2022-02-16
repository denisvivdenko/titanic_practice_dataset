import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from data_preprocessing import pipeline

data = pd.read_csv("train.csv")
X_train = pd.DataFrame(pipeline.fit_transform(data), index=data.PassengerId)
y_train = data.Survived

model = RandomForestClassifier(n_estimators=100, max_depth=30)
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
