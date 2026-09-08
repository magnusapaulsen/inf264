from sklearn.datasets import make_classification

def generate_synthetic_dataset(n_samples=100, n_features=19, random_state=67):
    X, y = make_classification(
        n_samples, 
        n_features,
        random_state
    )

    return X, y