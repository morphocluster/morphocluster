"""
pytest file for processing.prototypes.Prototypes
"""

import numpy as np
import pytest
from sklearn.cluster import MiniBatchKMeans

from morphocluster.processing.prototypes import Prototypes, merge_prototypes

N_FEATURES = 32


@pytest.fixture(params=[5, 10, 100], name="make_dset")
def fixture_make_dset(request):
    def _make_dset():
        return np.random.rand(request.param, N_FEATURES)

    return _make_dset


@pytest.fixture(params=[1, 5, 10, 100], name="k")
def fixture_k(request):
    return request.param


@pytest.fixture(params=[1, 5, 10], name="n_children")
def fixture_n_children(request):
    return request.param


def test_NCentroidDistance(make_dset, k):
    clusterer = MiniBatchKMeans(n_clusters=k)
    prots = Prototypes(clusterer)

    train = make_dset()
    print("train.shape", train.shape)

    if train.shape[0] == 0:
        with pytest.raises(ValueError):
            prots.fit(train)
        return

    prots.fit(train)

    assert prots.prototypes_.shape[0] <= k
    assert prots.prototypes_.shape[1] == N_FEATURES
    assert prots.prototypes_.shape[0] == prots.support_.shape[0]
    assert np.isfinite(prots.prototypes_).all()

    test = make_dset()
    distances = prots.transform(test)
    assert distances.shape[0] == test.shape[0]

    # The distances of the prototypes to themselves should be 0
    distances = prots.transform(prots.prototypes_)
    assert np.all(distances == 0)


def test_merge_prototypes(make_dset, k, n_children):
    children = []
    clusterer = MiniBatchKMeans(n_clusters=k)
    for _ in range(n_children):
        prots = Prototypes(clusterer)
        train = make_dset()

        try:
            prots.fit(train)
            children.append(prots)
        except ValueError:
            pass

    result = merge_prototypes(children, k)

    assert result.prototypes_.shape[0] <= k
    assert result.prototypes_.shape[1] == N_FEATURES
    assert result.prototypes_.shape[0] == result.support_.shape[0]
