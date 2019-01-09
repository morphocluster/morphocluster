import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.validation import check_is_fitted


class NCentroidDistance:
    """
    Calculate the distance of objects to a number of prototypes.

    Prototypes are calculated using sklearn.cluster.MiniBatchKMeans

    Parameters:
        k: int
            Number of prototypes
        clusterer_args: dict
            Additional arguments for MiniBatchKMeans

    Attributes:
        prototypes_: array of shape = [n_prototypes, n_features]
    """

    def __init__(self, k, clusterer_args=None):
        self.k = k
        self.clusterer_args = clusterer_args or {}

    def fit(self, X):
        """
        Calculate k prototypes based on X.

        Parameters:
            X: array of shape = [n_samples, n_features]
        """
        if self.k == 1:
            self.prototypes_ = np.mean(X, axis=0)[np.newaxis, :]
            return

        if X.shape[0] <= self.k:
            self.prototypes_ = X.copy()
            return

        clusterer = MiniBatchKMeans(self.k, **self.clusterer_args)
        clusterer.fit(X)
        self.prototypes_ = clusterer.cluster_centers_

    def transform(self, X, metric='euclidean', **kwargs):
        """
        Compute distance for every row in X.

        Uses scipy.spatial.distance.cdist.

        Parameters:
            X: array of shape = [n_samples, n_features]
            metric: str
                Metric for cdist
            **kwargs: dict
                Additional arguments for cdist
        """
        check_is_fitted(self, ["prototypes_"])

        # Calculate distance matrix
        dist_matrix = cdist(X, self.prototypes_, metric=metric, **kwargs)

        # Calculate minimum
        distances = np.min(dist_matrix, axis=1)

        return distances
