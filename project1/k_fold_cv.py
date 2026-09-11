import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

def split_dataset(dataset, folds):

    X, y = dataset
    subsets = []
    for k in range(folds):
        if k < folds-1:
            X, X_split, y, y_split = train_test_split(X, y, test_size=1/(folds-k))
            subsets.append((X_split, y_split))
        else:
            subsets.append((X, y))

    return subsets

d = [[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]]

subsets = split_dataset(d, 3)
print(subsets)