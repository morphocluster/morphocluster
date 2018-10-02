#!/usr/bin/env python3
import sys
import time

import fire
import h5py
import numpy as np
import pandas as pd

import hdbscan
from morphocluster.processing import Tree


class Recluster:
    def __init__(self):
        self.dataset = None
        self.trees = []

    def load_features(self, features_fn, append=True):
        """
        Load object features from a HDF5 file.

        Parameters:
            features_fn: HDF5 file.
            append: Append to the existing features (instead of replacing).
        """

        print("Loading {}...".format(features_fn))

        with h5py.File(features_fn, "r", libver="latest") as f_features:
            dataset = {t: f_features[t][:]
                       for t in ("features", "objids")}

        if append and self.dataset is not None:
            self.dataset["features"] = np.concatenate((self.dataset["features"],
                                                       dataset["features"]))

            self.dataset["objids"] = np.concatenate((self.dataset["objids"],
                                                     dataset["objids"]))
        else:
            self.dataset = dataset

        print("Loaded {:,d} features.".format(len(dataset["features"])))

        if append:
            print("Dataset size: {:,d}".format(len(self.dataset["features"])))

        return self

    def load_tree(self, in_tree_fn):
        """
        Load an existing cluster tree.
        """

        self.trees.append(Tree.from_saved(in_tree_fn))

        return self

    def _get_unapproved_dataset(self):
        approved_objids = []
        for i, tree in enumerate(self.trees):
            print("Tree #{}:".format(i))

            approved_nodes_selector = tree.nodes["approved"] == True
            approved_node_ids = tree.nodes.loc[approved_nodes_selector, "node_id"]

            approved_objects_selector = tree.objects["node_id"].isin(
                approved_node_ids)

            n_objects = len(tree.objects)
            n_approved_objects = approved_objects_selector.sum()

            approved_objids.append(
                tree.objects.loc[approved_objects_selector, "object_id"])

            print(" Approved objects: {:,d} / {:,d} ({:.2%})".format(
                n_approved_objects, n_objects, (n_approved_objects / n_objects)))
            print(" Unapproved objects: {:,d} / {:,d} ({:.2%})".format(
                n_objects - n_approved_objects, n_objects, ((n_objects - n_approved_objects) / n_objects)))

        approved_objids = pd.concat(
            approved_objids).drop_duplicates().reset_index(drop=True)

        # This is faster than np.isin
        dataset_objids = pd.Series(self.dataset["objids"])
        dataset_selector = ~dataset_objids.isin(approved_objids)

        n_selected = dataset_selector.sum()
        n_total = len(dataset_objids)

        print("Unapproved objects present in dataset: {:,d} / {:,d} ({:.2%})".format(
            n_selected, n_total, (n_selected / n_total)))

        return {
            "features": self.dataset["features"][dataset_selector].reset_index(drop=True),
            "objids": dataset_objids[dataset_selector].reset_index(drop=True)
        }

    def cluster(self, ignore_approved=True, **kwargs):
        """
        Cluster the data.
        """

        if ignore_approved:
            dataset = self._get_unapproved_dataset()
        else:
            dataset = self.dataset

        clusterer = hdbscan.HDBSCAN(**kwargs)

        print("Clustering...")
        start = time.perf_counter()
        labels = clusterer.fit_predict(dataset["features"])
        time_fit = time.perf_counter() - start

        print("Clustering took {:.0f}s".format(time_fit))

        # Turn cluster_labels to a tree
        self.trees.append(Tree.from_labels(labels, dataset["objids"]))

        return self

    def save(self, tree_fn):
        """
        Save the result combining all contained trees.
        """

        if not self.trees:
            raise ValueError("No trees.")

        tree = self.trees[0]

        for other_tree in self.trees[1:]:
            tree.merge(other_tree)

        tree.save(tree_fn)

        return self


if __name__ == "__main__":
    sys.exit(fire.Fire(Recluster()))
