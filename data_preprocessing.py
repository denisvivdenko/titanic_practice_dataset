import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from outliers_handler import OutliersIQRHandler
from bin_encoder import BinEncoder
from alphabetic_code_extractor import extract_alphabetic_code
from alphabetic_code_extractor import extract_alphabetic_code_by_split

data = pd.read_csv("train.csv")


age_feature_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="median")),
    ("outliers_handler", OutliersIQRHandler(strategy="median")),
    ("discretizer", BinEncoder(bins=[5, 12, 18, 25, 45, 60, 80]))
])

ticket_feature_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="most_frequent")),
    ("alphabetic_code_extractor", extract_alphabetic_code_by_split)
])

cabin_feature_pipeline = Pipeline([
    ("missing_values", SimpleImputer(strategy="most_frequent")),
    ("alphabetic_code_extractor", extract_alphabetic_code)
])


if __name__ == "__main__":
    print(age_feature_pipeline.fit_transform([data["Age"].values]))
    print(ticket_feature_pipeline.fit_transform([data["Ticket"].values]))
    print(cabin_feature_pipeline.fit_transform([data["Cabin"].values]))