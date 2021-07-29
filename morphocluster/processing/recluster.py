#!/usr/bin/env python3
import sys
import time
from typing import Any, Dict, List, Optional, Union

import fire
import h5py
import hdbscan
import numpy as np
import pandas as pd
import sklearn.decomposition

from morphocluster.processing import Tree


def _subsample_dataset(sample_size, dataset):
    """
    dataset["features"] is a numpy.ndarray
    dataset["object_id"] is a pandas.Series
    """

    features = dataset["features"]
    object_id = dataset["object_id"]

    if features.shape[0] <= sample_size:
        return dataset

    idx = np.random.permutation(features.shape[0])[:sample_size]

    features = features[idx]
    object_id = object_id.iloc[idx].reset_index(drop=True)

    return {"features": features, "object_id": object_id}


class Recluster:
    def __init__(self):
        self.dataset: Optional[Dict] = None
        self.trees: List[Tree] = []
        self.log: List = []

        self._log("initialize", dict(time=time.time()))

    def load_features(self, features_fn, append=True):
        """
        Load object features from a HDF5 file.

        Parameters:
            features_fn: HDF5 file.
            append: Append to the existing features (instead of replacing).
        """

        print("Loading {}...".format(features_fn))

        with h5py.File(features_fn, "r") as f_features:
            dataset = {
                "features": f_features["features"][:],
                # Sometimes, object_id are still ints (which is wrong)
                "object_id": pd.Series(f_features["object_id"].asstr()[:]),
            }

        if append and self.dataset is not None:
            self.dataset["features"] = np.concatenate(
                (self.dataset["features"], dataset["features"])
            )

            self.dataset["object_id"] = pd.concat(
                (self.dataset["object_id"], dataset["object_id"])
            ).reset_index(drop=True)

        else:
            self.dataset = dataset

        self._log("load_features", dict(append=append, features_fn=features_fn))

        print("Loaded {:,d} features.".format(len(dataset["features"])))

        if append:
            print("Dataset size: {:,d}".format(len(self.dataset["features"])))

        return self

    def load_tree(self, tree: Union[Tree, Any]):
        """
        Load an existing cluster tree.
        """

        if not isinstance(tree, Tree):
            print(f"Loading tree {tree}...")
            self._log("load_tree", dict(tree_fn=str(tree)))
            tree = Tree.from_saved(tree)
        else:
            self._log("load_tree")

        self.trees.append(tree)

        return self

    def _get_unapproved_dataset(self):
        approved_object_id = []
        tree_object_id = []

        # If no trees are loaded, the whole dataset is unapproved
        if not self.trees:
            return self.dataset

        for i, tree in enumerate(self.trees):
            print("Tree #{}:".format(i))

            approved_nodes_selector = tree.nodes["approved"] == True
            approved_node_ids = tree.nodes.loc[approved_nodes_selector, "node_id"]

            approved_objects_selector = tree.objects["node_id"].isin(approved_node_ids)

            n_objects = len(tree.objects)
            n_approved_objects = approved_objects_selector.sum()

            approved_object_id.append(
                tree.objects.loc[approved_objects_selector, "object_id"]
            )

            tree_object_id.append(tree.objects["object_id"])

            print(
                " Approved objects: {:,d} / {:,d} ({:.2%})".format(
                    n_approved_objects, n_objects, (n_approved_objects / n_objects)
                )
            )
            print(
                " Unapproved objects: {:,d} / {:,d} ({:.2%})".format(
                    n_objects - n_approved_objects,
                    n_objects,
                    ((n_objects - n_approved_objects) / n_objects),
                )
            )

        approved_object_id = (
            pd.concat(approved_object_id).drop_duplicates().reset_index(drop=True)
        )

        tree_object_id = (
            pd.concat(tree_object_id).drop_duplicates().reset_index(drop=True)
        )

        # This is faster than np.isin
        dataset_object_id = pd.Series(self.dataset["object_id"])
        dataset_available_selector = dataset_object_id.isin(tree_object_id)
        n_dataset_available = dataset_available_selector.sum()

        print(
            "Availability (dset/both/tree): {:,d} / {:,d} / {:,d}".format(
                len(dataset_available_selector) - n_dataset_available,
                n_dataset_available,
                len(tree_object_id) - n_dataset_available,
            )
        )

        dataset_selector = ~dataset_object_id.isin(approved_object_id)

        n_selected = dataset_selector.sum()
        n_total = len(dataset_object_id)

        print(
            "Unapproved objects present in dataset: {:,d} / {:,d} ({:.2%})".format(
                n_selected, n_total, (n_selected / n_total)
            )
        )

        return {
            "features": self.dataset["features"][dataset_selector],
            "object_id": dataset_object_id[dataset_selector].reset_index(drop=True),
        }

    def cluster(self, ignore_approved=True, sample_size=None, pca=None, **kwargs):
        """
        Cluster the data.
        """

        if ignore_approved:
            dataset = self._get_unapproved_dataset()
        else:
            dataset = self.dataset

        if sample_size is not None:
            print("Subsampling dataset ({:,d})...".format(sample_size))
            dataset = _subsample_dataset(sample_size, dataset)

        features = dataset["features"]
        if pca is not None:
            print(f"Performing PCA ({pca})...")
            start = time.perf_counter()
            pca = sklearn.decomposition.PCA(pca)
            features = pca.fit_transform(features)
            time_fit = time.perf_counter() - start
            print("Dimensionality reduction took {:.0f}s".format(time_fit))
            print("Explained variance ratio:", pca.explained_variance_ratio_.sum())

        print("Feature shape:", features.shape)
        print("Arguments:", kwargs)

        clusterer = hdbscan.HDBSCAN(**kwargs)

        n_objects = len(features)

        print(f"Clustering {n_objects:,d} objects...")
        start = time.perf_counter()
        labels = clusterer.fit_predict(features)
        time_fit = time.perf_counter() - start

        print("Clustering took {:.0f}s".format(time_fit))

        n_labels = len([l for l in np.unique(labels) if l != -1])
        print(f"Found {n_labels:,d} labels.")

        self._log(
            "cluster",
            dict(
                time_fit=time_fit,
                ignore_approved=ignore_approved,
                sample_size=sample_size,
                kwargs=kwargs,
                n_objects=n_objects,
                n_labels=n_labels,
            ),
        )

        # Turn cluster_labels to a tree
        self.trees.append(Tree.from_labels(labels, dataset["object_id"]))

        return self

    def save_all(self, prefix):
        """
        Save all trees individually.
        """
        for i, tree in enumerate(self.trees):
            tree_fn = "{}-{:d}.zip".format(prefix, i)
            tree.save(tree_fn)

        return self

    def merge_trees(self) -> Tree:
        if not self.trees:
            raise ValueError("No trees.")

        tree = self.trees[0].copy()

        for other_tree in self.trees[1:]:
            tree.merge(other_tree)

        return tree

    def save(self, tree_fn):
        """
        Save the result combining all contained trees.
        """

        tree = self.merge_trees()
        tree.save(tree_fn, dict(log=self.log))

        print(f"Saved tree to {tree_fn}.")

        return self

    def stop(self):
        self._log("stop", dict(time=time.time()))

    def _log(self, topic, data=None):
        self.log.append({topic: data})


if __name__ == "__main__":
    sys.exit(fire.Fire(Recluster()))
