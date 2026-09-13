from sklearn.inspection import permutation_importance

def permutation_importance(model, X, y, metric, n_repeats, seed):
    r = permutation_importance(model, X, y, metric, n_repeats, seed)

