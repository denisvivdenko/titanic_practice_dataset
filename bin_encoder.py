from typing import List

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np

class BinEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, bins: List[int]) -> None:
        """
        params:
            bins (List[int]): bins by right border
        """
        self.bins = bins
    
    def fit(self, X: np.array, y=None):
        return self
    
    def transform(self, X: np.array) -> np.array:
        """
        Divide array on bins

        params:
            X (np.array): processing data

        returns:
            (np.array) one dimensional discretized data
        """
        return np.digitize(X, bins=self.bins, right=True)


if __name__ == "__main__":
    bin_encoder = BinEncoder(bins=[25, 50, 75, 100])
    X = np.arange(0, 120, dtype=int)
    print(X, bin_encoder.fit_transform(X))