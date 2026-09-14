def permutation_importance(model, X, y, metric, n_repeats, seed):
    """
    For each feature in the dataset, we shuffle a features values n-times, measuring the performance drop, then move on to the next feature.
    We use accuracy as our metric.
    We test it on our test dataset. This let's us see which features are important for generalization, as opposed to checking on training data.
    We should use n_repeats=30
    (importance)
    i = s - 1/k sum[1,k](sk)
    Our real accuracy score minus the mean accuracy score of the permuted version
    This is the importance of a feature. We do this for all features.
    """
    # Metric is accuracy
    # Use test data
    # 30 repeats
    
    pass