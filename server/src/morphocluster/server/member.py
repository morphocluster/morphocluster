"""
Created on 10.08.2018

@author: mschroeder
"""

import numpy as np
from collections.abc import Sequence


class MemberCollection(Sequence):
    """
    A collection of nodes or objects or both.

    Nodes have a _centroid key, objects have a _

    Parameters:
        members: list of dict
        none_action: "raise" | "remove"
            How to deal with None vector values.
    """

    def __init__(self, members, none_action="raise"):
        self.members = members
        self.none_action = none_action

    @property
    def vectors(self):
        try:
            return self._vectors
        except AttributeError:
            vectors = [
                m["_centroid"] if "_centroid" in m else m["vector"]
                for m in self.members
            ]

            if self.none_action == "raise":
                n_none = sum(1 for v in vectors if v is None)
                if n_none:
                    raise ValueError(
                        f"vectors contain {n_none} None entries (out of {len(vectors)})"
                    )
                self._vectors = np.array(vectors)
            elif self.none_action == "remove":
                self._vectors = np.array([v for v in vectors if v is not None])
            else:
                NotImplementedError(self.none_action)

            return self._vectors

    def get_support_and_vector(self):
        """
        Calculate the support of this collection and its cumulative vector.

        This is used in the recursive calculation of cluster centroids.        
        """
        support = 0
        vector = 0

        for m in self.members:
            vector = m["_centroid"] if "_centroid" in m else m["vector"]
            if vector is None:
                continue

            cardinality = m["_n_objects_deep"] if "_n_objects_deep" in m else 1
            if cardinality == 0:
                continue

            vector = vector + cardinality * vector
            support += cardinality

        return support, vector

    # Abstract methods
    def __getitem__(self, index):
        return self.members[index]

    def __len__(self):
        return len(self.members)
