import numpy as np
from sklearn.preprocessing import FunctionTransformer

@FunctionTransformer
def extract_alphabetic_code_by_split(X: np.array, split_value: str = " ") -> np.array:
    extract_code = np.vectorize(lambda x: " ".join([code for code in x.split(split_value) if not code.isnumeric()]))
    return extract_code(X)

@FunctionTransformer
def extract_alphabetic_code(X: np.array) -> np.array:
    extract_alphabetic = np.vectorize(lambda x: ''.join(filter(str.isalpha, str(x))))
    return extract_alphabetic(X)


if __name__ == "__main__":
    print(extract_alphabetic_code_by_split.fit_transform(np.array(["B1C 12", "AB 3", "K 1"])))
    print(extract_alphabetic_code.fit_transform(np.array(["B1C 12", "AB 3B", "K 121C"])))

"""
input: ["B1C 12", "AB 3", "K 1"]
output: ['B1C' 'AB' 'K']

input: ["B1C 12", "AB 3B", "K 121C"]
output: ['BC' 'ABB' 'KC']
"""
