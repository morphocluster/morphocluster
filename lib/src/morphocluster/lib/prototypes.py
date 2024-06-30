import numpy as np
from numpy.lib.arraysetops import unique
from scipy.spatial.distance import cdist
from sklearn.base import ClassifierMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_is_fitted


def _check_is_clusterer(clusterer):
    attributes = ("fit_predict", "n_clusters")
    if not all(hasattr(clusterer, attr) for attr in attributes):
        raise TypeError("%s is not a proper clusterer instance." % (clusterer))


class Prototypes:
    """
    Represent a number of vectors by a lower number of prototypes.

    Prototypes are calculated using sklearn.cluster.MiniBatchKMeans.

    Parameters:
        k: int
            Number of prototypes
        clusterer_args: dict
            Additional arguments for MiniBatchKMeans

    Attributes:
        prototypes_: array of shape = [n_prototypes, n_features]
        support_: array of shape = [n_prototypes]
    """

    def __init__(self, clusterer):
        self.clusterer = clusterer

    def fit(self, X):
        """
        Calculate k prototypes based on X.

        Parameters:
            X: array of shape = [n_samples, n_features]
        """

        _check_is_clusterer(self.clusterer)

        if X.shape[0] == 0:
            raise ValueError("X contains no samples.")

        # If only one centroid, calculate mean
        if self.clusterer.n_clusters == 1:
            self.prototypes_ = np.mean(X, axis=0)[np.newaxis, :]
            self.support_ = np.array([X.shape[0]])
            self._validate()

            return

        # If fewer data points than centroids, use them
        if X.shape[0] <= self.clusterer.n_clusters:
            self.prototypes_ = X.copy()
            self.support_ = np.ones(X.shape[0])
            self._validate()

            return

        labels = self.clusterer.fit_predict(X)

        self.prototypes_ = self.clusterer.cluster_centers_
        self.support_ = np.bincount(labels, minlength=self.prototypes_.shape[0])
        self._validate()

    def _validate(self):
        """Validate that prototypes and support match."""
        assert self.prototypes_.shape[0] == self.support_.shape[0], f"Prototype shape ({self.prototypes_.shape}) does not support shape ({self.support_.shape})"

    def transform(self, X, metric="euclidean", **kwargs):
        """
        Compute distance for every row in X.

        Uses scipy.spatial.distance.cdist.

        Parameters:
            X: array of shape = [n_samples, n_features]
            metric: str
                Metric for cdist
            **kwargs: dict
                Additional arguments for cdist
        Returns: array of shape = [n_samples] or [n_samples, n_labels]

        TODO: Accept a list of Prototypes as X
        """
        check_is_fitted(self, ["prototypes_", "support_"])

        if self.prototypes_.shape[0] == 0:
            return np.zeros(X.shape[0]) + np.inf

        if X.shape[0] == 0:
            return np.zeros(0) + np.inf

        # Calculate distance matrix
        dist_matrix = cdist(X, self.prototypes_, metric=metric, **kwargs)

        # Calculate minimum
        distances = np.min(dist_matrix, axis=1)

        return distances

    def distance(self, other, metric="euclidean", **kwargs):
        """
        Compute the distance of self to another Prototypes object.

        Modified Hausdorff Distance (MHD) according to
            Dubuisson, M.-P. and Jain, A. K. (1994) ‘A modified Hausdorff distance for object matching’,
            in Proceedings of 12th International Conference on Pattern Recognition.
            IEEE Comput. Soc. Press, pp. 566–568. doi: 10.1109/ICPR.1994.576361.
        """
        dist_matrix = cdist(other.prototypes, self.prototypes_, metric=metric, **kwargs)

        return max(np.mean(np.min(dist_matrix, axis=a)) for a in (0, 1))


class PrototypeClassifier(ClassifierMixin):
    def __init__(self, clusterer, metric="euclidean", n_classes=None):
        self.metric = metric
        self.clusterer = clusterer
        self.n_classes = n_classes

    def fit(self, X, y):
        """
        Parameters:
            X (ndarray of shape = [n_samples, n_features]): Feature vectors
            y (ndarray of shape = [n_samples]): Labels in the range(0, n_classes)
        """
        # Fit prototypes
        self.prototypes_ = {}

        if self.n_classes is None:
            self.n_classes = np.max(y) + 1

        for label in range(self.n_classes):
            mask = y == label

            prototypes = Prototypes(self.clusterer)

            try:
                prototypes.fit(X[mask])
            except ValueError:
                pass

            self.prototypes_[label] = prototypes

    def predict_score(self, X, _softmax=True):
        check_is_fitted(self, "prototypes_")

        distances = np.inf + np.zeros(
            (X.shape[0], max(self.prototypes_.keys()) + 1), dtype=X.dtype
        )

        for label, prototypes in self.prototypes_.items():
            try:
                distances[:, label] = prototypes.transform(X, self.metric)
            except NotFittedError:
                distances[:, label] = np.inf

        if _softmax:
            return softmax(-distances)

        return -distances

    def predict(self, X):
        proba = self.predict_score(X, _softmax=True)

        return np.argmax(proba, axis=1)

    def mse(self, X, y):
        """
        TODO: Calculate the MSE of samples in X corresponding to class y.
        """
        ...


def merge_prototypes(children, k, metric="euclidean"):
    """
    Merge a number of prototypes of a list of children so that k new prototypes result.

    Parameters:
        children: list of Prototypes objects
        k: int
            Number of resulting prototypes

    Returns: Prototypes object
    """
    clusterer = AgglomerativeClustering(
        n_clusters=k, affinity=metric, linkage="complete"
    )

    if not children:
        return Prototypes(None)

    for c in children:
        check_is_fitted(c, ["prototypes_", "support_"])

    prototypes_ = np.concatenate([ncd.prototypes_ for ncd in children])
    support_ = np.concatenate([ncd.support_ for ncd in children])

    if prototypes_.shape[0] <= k:
        result = Prototypes(k)
        result.prototypes_ = prototypes_
        result.support_ = support_

        return result

    labels = clusterer.fit_predict(prototypes_)
    unique_labels = np.unique(labels)

    new_prototypes = np.empty(
        (unique_labels.shape[0], prototypes_.shape[1]), prototypes_.dtype
    )
    new_support = np.empty((unique_labels.shape[0]), support_.dtype)

    for label in unique_labels:
        mask = labels == label
        new_support[label] = support_[mask].sum()

        # Weighted sum of all elements in the cluster
        x = np.sum(prototypes_[mask] * support_[mask, np.newaxis], axis=0)
        # Normalize
        x /= new_support[label]

        new_prototypes[label] = x

    result = Prototypes(None)
    result.prototypes_ = new_prototypes
    result.support_ = new_support

    return result
