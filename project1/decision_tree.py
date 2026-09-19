"""
Decision tree with Iterative Dichotomizer 3 (ID3) learning algorithm.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class Node:
    """ This is the class definition of nodes that make up my decision tree """
    def __init__(self, feature, threshold, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        """ Checks if a node is a leaf node or not, returns True or False """
        return self.value is not None
        

class DecisionTree(BaseEstimator, ClassifierMixin):
    """ This is the class definition for my model """
    def __init__(self, criterion="entropy", max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth

    def _split(self, X, y, depth):
        """ Helper function which actually does the splitting """
        # Checking base case
        if depth >= self.max_depth_ or len(np.unique(y)) == 1 or np.all(X == X[0, :]):
            node = Node(feature=None, threshold=None)
            values, counts = np.unique(y, return_counts=True)
            node.value = values[np.argmax(counts)]
            return node

        # Finding best feature and threshold to split on
        best_IG = -np.inf # Negative infinity since I am comparing
        best_feature = None
        best_threshold = None

        features = range(X.shape[1])
        for feature in features:
            median = np.median(X[:, feature])

            left_mask = X[:, feature] <= median
            right_mask = X[:, feature] > median

            X_left, y_left = X[left_mask], y[left_mask]
            X_right, y_right = X[right_mask], y[right_mask]

            IG = self.information_gain(y, y_left, y_right)
            if IG > best_IG:
                best_IG = IG
                best_feature = feature
                best_threshold = median

        # Check for 0 information gain
        if best_IG <= 0:
            node = Node(feature=None, threshold=None)
            values, counts = np.unique(y, return_counts=True)
            node.value = values[np.argmax(counts)]
            return node

        # Splitting the dataset into left and right
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # Building the node
        node = Node(best_feature, best_threshold)

        # Recursing into left and right
        node.left = self._split(X_left, y_left, depth + 1)
        node.right = self._split(X_right, y_right, depth + 1)

        # Finally, returning this subtree
        return node

    def fit(self, X, y):
        """ Kicks off the building of the decision tree """
        self.max_depth_ = np.inf if self.max_depth is None else self.max_depth
        self.classes_ = np.unique(y)
        self.root = self._split(X, y, 0)

        return self

    def predict(self, X):
        """ Takes a dataset, X, as input and returns predicted labels, y """
        return np.array([self._traverse(x, self.root) for x in X])


    def _traverse(self, x, node):
        """ Helper function to traverse the tree for prediction """
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def show_tree(self, node=None, depth=0, feature_names=None):
        """ Visualize the tree """
        if node is None:
            node = self.root

        indent = "  " * depth
        if node.is_leaf():
            print(f"{indent}Predict: {node.value}")
            return

        feature_name = feature_names[node.feature] if feature_names is not None else node.feature
        print(f"{indent}{feature_name} <= {node.threshold:.2f}?")
        print(f"{indent}Left:")
        self.show_tree(node.left, depth+1, feature_names=feature_names)
        print(f"{indent}Right:")
        self.show_tree(node.right, depth+1, feature_names=feature_names)

    def entropy(self, y):
        """ H(y) """
        _, counts = np.unique(y, return_counts=True)
        p = counts/len(y)
        return -np.sum(p*np.log2(p))

    def conditional_entropy(self, y_left, y_right):
        """ H(y|x) """
        h_left = self.entropy(y_left)
        h_right = self.entropy(y_right)

        lyl, lyr = len(y_left), len(y_right)
        ly = lyl + lyr

        return (lyl/ly * h_left + (lyr/ly) * h_right)

    def gini(self, y):
        """ G(y) """
        _, counts = np.unique(y, return_counts=True)
        p = counts/len(y)
        return np.sum(p*(1-p))

    def conditional_gini(self, y_left, y_right):
        """G(y|x)"""
        g_left = self.gini(y_left)
        g_right = self.gini(y_right)

        lyl, lyr = len(y_left), len(y_right)
        ly = lyl + lyr

        return (lyl/ly * g_left + (lyr/ly) * g_right)

    def information_gain(self, y, y_left, y_right):
        """ IG = H(y) - H(y|x) or IG = G(y) - G(y|x) """
        if self.criterion == "gini":
            return self.gini(y) - self.conditional_gini(y_left, y_right)
        return self.entropy(y) - self.conditional_entropy(y_left, y_right)   

if __name__ == '__main__':
    # This was just used for testing
    from synthetic_dataset import generate_synthetic_dataset
    X, y = generate_synthetic_dataset(
        n_samples=100,
        n_features=19,
        weights=[0.74,0.16],
        random_state=67
    )

    dt = DecisionTree(criterion="entropy", max_depth=3)
    dt.fit(X, y)
    dt.show_tree()
    y_pred = dt.predict(X)

    from sklearn.metrics import f1_score, fbeta_score, balanced_accuracy_score

    print(f"F1-Score: {f1_score(y, y_pred)}")
    print(f"F0.5-Score: {fbeta_score(y, y_pred, beta=0.5)}")
    print(f"F2-Score: {fbeta_score(y, y_pred, beta=2)}")
    print(f"Balanced Accuracy Score: {balanced_accuracy_score(y, y_pred)}")