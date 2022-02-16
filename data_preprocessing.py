import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from outliers_handler import OutliersIQRHandler
from bin_encoder import BinEncoder

data = pd.read_csv("train.csv")

age_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="median")),
    ("outliers_handler", OutliersIQRHandler(strategy="median")),
    ("discretizer", BinEncoder(bins=[5, 12, 18, 25, 45, 60, 80]))
])

print(age_pipeline.fit_transform([data["Age"].values]))