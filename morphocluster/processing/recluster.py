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

        with h5py.File(features_fn, "r") as f_features:
            dataset = {
                "features": f_features["features"][:],
                # Sometimes, objids are still ints (which is wrong)
                "objids": pd.Series(f_features["objids"][:]).astype(str)
            }

        if append and self.dataset is not None:
            self.dataset["features"] = np.concatenate((self.dataset["features"],
                                                       dataset["features"]))

            self.dataset["objids"] = pd.concat((self.dataset["objids"],
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
        tree_objids = []
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

            tree_objids.append(tree.objects["object_id"])

            print(" Approved objects: {:,d} / {:,d} ({:.2%})".format(
                n_approved_objects, n_objects, (n_approved_objects / n_objects)))
            print(" Unapproved objects: {:,d} / {:,d} ({:.2%})".format(
                n_objects - n_approved_objects, n_objects, ((n_objects - n_approved_objects) / n_objects)))

        approved_objids = pd.concat(
            approved_objids).drop_duplicates().reset_index(drop=True)

        tree_objids = pd.concat(
            tree_objids).drop_duplicates().reset_index(drop=True)

        # This is faster than np.isin
        dataset_objids = pd.Series(self.dataset["objids"])
        dataset_available_selector = dataset_objids.isin(tree_objids)
        n_dataset_available = dataset_available_selector.sum()

        print("Availability (dset/both/tree): {:,d} / {:,d} / {:,d}".format(
            len(dataset_available_selector) - n_dataset_available,
            n_dataset_available,
            len(tree_objids) - n_dataset_available))

        dataset_selector = ~dataset_objids.isin(approved_objids)

        n_selected = dataset_selector.sum()
        n_total = len(dataset_objids)

        print("Unapproved objects present in dataset: {:,d} / {:,d} ({:.2%})".format(
            n_selected, n_total, (n_selected / n_total)))

        return {
            "features": self.dataset["features"][dataset_selector],
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

        print("Clustering {:,d} objects...".format(len(dataset["features"])))
        start = time.perf_counter()
        labels = clusterer.fit_predict(dataset["features"])
        time_fit = time.perf_counter() - start

        print("Clustering took {:.0f}s".format(time_fit))

        print("Found {:,d} labels.".format(len(np.unique(labels))))

        # Turn cluster_labels to a tree
        self.trees.append(Tree.from_labels(labels, dataset["objids"]))

        return self

    def save_all(self, prefix):
        """
        Save all trees individually.
        """
        for i, tree in enumerate(self.trees):
            tree_fn = "{}-{:d}.zip".format(prefix, i)
            tree.save(tree_fn)

        return self

    def save(self, tree_fn):
        """
        Save the result combining all contained trees.
        """

        if not self.trees:
            raise ValueError("No trees.")

        tree = self.trees[0].copy()

        for other_tree in self.trees[1:]:
            tree.merge(other_tree)

        tree.save(tree_fn)

        return self


if __name__ == "__main__":
    sys.exit(fire.Fire(Recluster()))
