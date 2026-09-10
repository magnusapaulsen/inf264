from sklearn.datasets import make_classification

def generate_synthetic_dataset(n_samples=100, n_features=19, weights=[0.74, 0.16], random_state=67):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, weights=weights, random_state=random_state)

    return X, y