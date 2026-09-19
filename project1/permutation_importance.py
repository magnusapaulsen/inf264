import numpy as np

def permutation_importance(model, X, y, metric, n_repeats, seed):
    rng = np.random.default_rng(seed=seed)
    y_pred = model.predict(X)
    reference_score = metric(y, y_pred)

    importances = []

    for j in range(len(X[0])):
        importances_j = []
        for k in range(n_repeats):
            X_copy = X.copy()
            rng.shuffle(X_copy[:, j])
            y_pred = model.predict(X_copy)
            score_jk = metric(y, y_pred)
            importance = reference_score - score_jk
            importances_j.append(importance)

        importances.append(sum(importances_j)/len(importances_j))
    
    return np.array(importances)