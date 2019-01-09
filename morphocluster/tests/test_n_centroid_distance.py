"""
pytest file for processing.n_centroid_distance.NCentroidDistance
"""

import numpy as np
from pytest import fixture

from morphocluster.processing.n_centroid_distance import NCentroidDistance

N_FEATURES = 32


@fixture(params=[5, 10, 100], name="make_dset")
def fixture_make_dset(request):
    def _make_dset():
        return np.random.rand(request.param, N_FEATURES)

    return _make_dset


@fixture(params=[1, 5, 10, 100], name="k")
def fixture_k(request):
    return request.param


def test_NCentroidDistance(make_dset, k):
    dist = NCentroidDistance(k)
    train = make_dset()
    dist.fit(train)

    assert dist.prototypes_.shape[0] <= k
    assert dist.prototypes_.shape[1] == N_FEATURES

    test = make_dset()
    distances = dist.transform(test)
    assert distances.shape[0] == test.shape[0]

    # The distances of the prototypes themselves should be 0
    distances = dist.transform(dist.prototypes_)
    print(distances)
    assert np.all(distances == 0)
