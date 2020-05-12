"""a
Created on 27.03.2018

@author: mschroeder
"""

import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist


class Classifier(object):
    """
    This classifier assumes that all vectors are scaled to unit length.
    """

    def __init__(self, X):
        self.types = X

        # Calculate radius for each type
        distances = squareform(pdist(X))
        np.fill_diagonal(distances, np.inf)
        self.radius = np.min(distances, axis=0)

    def distances(self, X):
        """
        Calculates distances between types and X.
        
        Parameters:
            X: ndarray of N samples by M dimensions.
        
        Returns:
            ndarray of distance matrix
        """
        distances = cdist(self.types, X)

        return distances

    def classify(self, X, safe=True):
        """
        Classifies X into the types.
        
        Parameters:
            X: ndarray of N samples by D dimensions.
        
        Returns:
            ndarray of N type indices. -1 for unclassified 
        """

        distances = self.distances(X)

        min_dist_idx = np.argmin(distances, axis=0)

        min_dist = distances[min_dist_idx, np.arange(X.shape[0])]

        if safe:
            threshold = self.radius[min_dist_idx]
            return np.where(min_dist < threshold, min_dist_idx, -1)
        return min_dist_idx


if __name__ in ("builtins", "__main__"):
    n_starred = 100
    n_unstarred = 5000
    n_dim = 32

    starred = np.random.rand(n_starred, n_dim) - 0.5
    starred /= np.linalg.norm(starred, axis=1)[:, None]

    unstarred = np.random.rand(n_unstarred, n_dim) - 0.5
    unstarred /= np.linalg.norm(unstarred, axis=1)[:, None]

    distances_eucl = squareform(pdist(starred))
    np.fill_diagonal(distances_eucl, np.inf)
    min_rad_eucl = np.min(distances_eucl, axis=0)

    distances_cos = squareform(pdist(starred, "cosine"))
    np.fill_diagonal(distances_cos, np.inf)
    min_rad_cos = np.min(distances_cos, axis=0)

    assert np.all(np.argsort(min_rad_eucl) == np.argsort(min_rad_cos))
