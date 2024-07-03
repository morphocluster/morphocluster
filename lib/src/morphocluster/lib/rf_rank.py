"""
RF rank.
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np


def rf_rank(candidates, valid, rejected, **kwargs):
    """
    Rank recommendations using random forests.

    A model is trained to tell the level of membership of candidate vectors to
    a group of valid vectors given a set of already rejected vectors.

    Parameters:
        candidates: (N,D) np.array of N feature vectors of dimensionality D that are to be ranked.
        valid: (:,D) np.array of feature vectors that are valid members.
        rejected: (:,D) np.array of feature vectors that where already rejected.

        kwargs: Passed to RandomForestClassifier.
    """

    assert candidates.shape[0] > 0
    assert valid.shape[0] > 0
    assert rejected.shape[0] > 0

    classifier = RandomForestClassifier(**kwargs)

    X = np.concatenate((rejected, valid))
    y = np.ones(X.shape[0])
    y[: rejected.shape[0]] = -1

    classifier.fit(X, y)

    # Get index of positive class
    positive_index = np.nonzero(classifier.classes_ == 1)[0][0]

    proba = classifier.predict_proba(candidates)[:, positive_index]

    order = np.argsort(proba)[::-1]
    return order
