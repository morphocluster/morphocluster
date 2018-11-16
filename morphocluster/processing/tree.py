#!/usr/bin/env python3
"""
Conversion between different tree formats.
"""

import os
import sys
from io import StringIO
from zipfile import ZIP_DEFLATED, ZipFile

import fire
import numpy as np
import pandas as pd


class Tree(object):
    """
    Conversion between different tree formats.

    Members
        nodes: pandas.DataFrame
            node_id (int): ID of the current node
            parent_id (int): ID of the parent
            <Additional fields>
        objects: pandas.DataFrame
            object_id (str): Object identifier
            node_id (int): ID of the corresponding node
        rejected_objects: pandas.DataFrame (optional)
            object_id (str): Object identifier
            node_id (int): ID of the corresponding node
    """

    @staticmethod
    def from_collection(collection_fn, unlabeled_collection_fn=None):
        """
        Read a collection of objects.
        """
        collection = pd.read_csv(collection_fn,
                                 header=None,
                                 names=("objid", "path", "label"),
                                 index_col=False,
                                 usecols=("objid", "label",),
                                 dtype=str)

        grouped = collection.groupby("label")

        root_id = len(grouped)

        nodes = []
        objects = []

        nodes.append({"node_id": root_id})

        for i, (label, group) in enumerate(grouped):
            nodes.append({"node_id": i, "name": label, "parent_id": root_id})
            objects.extend({"object_id": objid, "node_id": i}
                           for objid in group["objid"])

        nodes = pd.DataFrame(nodes)

        objects = pd.DataFrame(objects)

        if unlabeled_collection_fn is not None:
            unlabeled_collection = pd.read_csv(unlabeled_collection_fn,
                                               header=None,
                                               names=(
                                                   "object_id", "path", "label"),
                                               index_col=False,
                                               usecols=("object_id",),
                                               dtype=str)

            # TODO: We blindly assume that collection and unlabeled collection are distinct
            unlabeled_collection["node_id"] = root_id

            objects = pd.concat(
                [objects, unlabeled_collection], ignore_index=True)

        return Tree(nodes, objects)

    @staticmethod
    def from_saved(tree_fn):
        """
        Read a saved tree.
        """
        with ZipFile(tree_fn, "r") as archive:
            with archive.open("nodes.csv", "r") as nodes_f:
                nodes = pd.read_csv(nodes_f)
            with archive.open("objects.csv", "r") as objects_f:
                objects = pd.read_csv(objects_f, dtype={'object_id': str})

            try:
                with archive.open("rejected_objects.csv", "r") as objects_f:
                    rejected_objects = pd.read_csv(objects_f, dtype={'object_id': str})
            except KeyError:
                # No such member
                rejected_objects = None

        return Tree(nodes, objects, rejected_objects)

    @staticmethod
    def from_HDBSCAN(path, root_first=True):
        """
        Read the tree from a HDBSCAN clustering.
        tree.csv is the condensed tree.
        objids.csv are object IDs in the same ordering as in the condensed tree.
        """

        tree_fn = os.path.join(path, "tree.csv")
        objids_fn = os.path.join(path, "objids.csv")

        raw_tree = pd.read_csv(tree_fn,
                               index_col=False,
                               dtype={
                                   "parent": np.uint64,
                                   "child": np.uint64,
                                   "lambda_val": np.float64,
                                   "child_size": np.uint64,
                                   "name": str
                               })

        objects = pd.read_csv(objids_fn, index_col=False, names=[
            "object_id"], header=None)

        raw_tree_nodes = raw_tree[raw_tree["child_size"] > 1].sort_values(
            "parent", ascending=root_first)
        raw_tree_objects = raw_tree[raw_tree["child_size"] == 1]

        root_id = raw_tree_nodes["parent"].iloc[0]

        # Get object ids for the root
        object_idxs = raw_tree_objects[raw_tree_objects["parent"]
                                       == root_id]["child"]
        root_objects = objects[object_idxs]
        root_objects["node_id"] = root_id

        nodes = raw_tree_nodes.rename(columns={"parent": "parent_id",
                                               "child": "node_id"})
        nodes = nodes.append({"node_id": root_id}, ignore_index=True)

        objects = pd.merge(objects, raw_tree_objects,
                           left_index=True, right_on="child")
        objects = objects[["object_id", "parent"]].rename(
            columns={"parent": "node_id"})

        objects = pd.concat([objects, root_objects], ignore_index=True)

        return Tree(nodes, objects)

    @staticmethod
    def from_labels(labels, object_ids):
        """
        Construct a tree from a label vector and a vector of object_ids.

        labels may contain -1. Then an object is assigned to root.
        """

        labels = pd.Series(labels)
        object_ids = pd.Series(object_ids)

        unique_labels = labels.unique()
        root_id = unique_labels.max() + 1

        # Compose nodes out of unique_labels (except -1)
        nodes = [{"node_id": node_id, "parent_id": root_id}
                 for node_id in unique_labels if node_id != -1]
        # Add root node
        nodes.append({"node_id": root_id})
        nodes = pd.DataFrame(nodes)

        # Compose objects
        objects = pd.concat({"node_id": labels, "object_id": object_ids}, axis=1)
        # Mount unlabeled objects (-1) to root
        objects.loc[objects["node_id"] == -1, "node_id"] = root_id

        return Tree(nodes, objects)

    @staticmethod
    def from_cluster_labels(cluster_labels_fn, object_ids_fn=None):
        """
        Construct tree from a cluster_labels file.

        A cluster_labels contains the columns "objid" and "label".
        """
        cluster_labels = pd.read_csv(cluster_labels_fn, index_col=False)

        unique_labels = cluster_labels["label"].unique()

        root_id = unique_labels.max() + 1

        nodes = [{"node_id": node_id, "parent_id": root_id}
                 for node_id in unique_labels]
        nodes.append({"node_id": root_id})

        nodes = pd.DataFrame(nodes)

        objects = cluster_labels.rename(
            columns={"objid": "object_id", "label": "node_id"})

        if object_ids_fn is not None:
            all_objects = pd.read_csv(object_ids_fn, index_col=False, names=[
                                      "object_id"], header=None)

            # Select all rows where an object_id is not in objects["object_id"]
            selector = ~all_objects["object_id"].isin(objects["object_id"])
            root_objects = all_objects[selector].copy()
            root_objects["node_id"] = root_id

            objects = pd.concat([objects, root_objects], ignore_index=True)

        return Tree(nodes, objects)

    def __init__(self, nodes=None, objects=None, rejected_objects=None):
        if nodes is not None:
            if not isinstance(nodes, pd.DataFrame):
                nodes = pd.DataFrame(nodes)

            if not "node_id" in nodes.columns:
                raise ValueError("'nodes' lacks column 'node_id'.")
            if not "parent_id" in nodes.columns:
                raise ValueError("'nodes' lacks column 'parent_id'.")

        if objects is not None:
            if not isinstance(objects, pd.DataFrame):
                objects = pd.DataFrame(objects)

            if not "object_id" in objects.columns:
                raise ValueError("'objects' lacks column 'object_id'.")
            if not "node_id" in objects.columns:
                raise ValueError("'objects' lacks column 'node_id'.")

        if rejected_objects is not None:
            if not isinstance(rejected_objects, pd.DataFrame):
                rejected_objects = pd.DataFrame(rejected_objects)

            if not "object_id" in rejected_objects.columns:
                raise ValueError("'rejected_objects' lacks column 'object_id'.")
            if not "node_id" in rejected_objects.columns:
                raise ValueError("'rejected_objects' lacks column 'node_id'.")

        self.nodes = nodes
        self.objects = objects
        self.rejected_objects = rejected_objects

    def save(self, tree_fn):
        """
        Save nodes and objects to an archive.
        """
        with ZipFile(tree_fn, "w", ZIP_DEFLATED) as archive:
            buffer_ = StringIO()
            self.nodes.to_csv(buffer_, index=False)
            archive.writestr("nodes.csv", buffer_.getvalue())

            buffer_ = StringIO()
            self.objects.to_csv(buffer_, index=False)
            archive.writestr("objects.csv", buffer_.getvalue())

            if self.rejected_objects is not None:
                buffer_ = StringIO()
                self.rejected_objects.to_csv(buffer_, index=False)
                archive.writestr("rejected_objects.csv", buffer_.getvalue())

    def get_root_id(self):
        """
        Get the ID of the root node.
        """
        selector = self.nodes["parent_id"].isnull()
        return np.asscalar(self.nodes.loc[selector, "node_id"])

    def topological_order(self):
        """
        Yield nodes in topological order.
        """

        for node_idx in self.topological_order_idx():
            yield self.nodes.loc[node_idx]

    def topological_order_idx(self):
        """
        Yield node indices in topological order.
        """
        queue = [self.get_root_id()]

        while queue:
            node_id = queue.pop()

            child_selector = self.nodes["parent_id"] == node_id
            queue.extend(self.nodes.loc[child_selector, "node_id"])

            node_selector = self.nodes["node_id"] == node_id

            if not node_selector.any():
                raise ValueError(
                    "No matching row for node_id={}".format(node_id))

            node_idx = node_selector.idxmax()

            yield node_idx

    def objects_for_node(self, node_id):
        """
        Return the objects of a certain node.
        """
        object_selector = self.objects["node_id"] == node_id
        return self.objects[object_selector]

    def print_topological_order(self):
        """
        Print nodes in topological order.
        """
        for node in self.topological_order():
            print(node)

    def print_objects_for_node(self, node_id):
        """
        Print objects of a certain node.
        """
        print(self.objects_for_node(node_id)["object_id"].tolist())

    def merge(self, other):
        """
        Merge other into self.

        If `other` contains the objects also present in `self`, their position in `other` takes precedence.
        """

        other_nodes = other.nodes.copy()
        other_objects = other.objects.copy()
        other_root = other.get_root_id()
        self_root = self.get_root_id()

        # Offset `node_id`s of other so they come after the `node_id`s of self
        offset = self.nodes["node_id"].max() + 1 - other_nodes["node_id"].min()
        other_nodes["node_id"] += offset
        other_nodes["parent_id"] += offset
        other_objects["node_id"] += offset
        other_root += offset

        # Delete other.root and relocate children and objects to self.root
        other_nodes = other_nodes[other_nodes["node_id"] != other_root]
        other_nodes.loc[other_nodes["parent_id"] == other_root, "parent_id"] = self_root
        other_objects.loc[other_objects["node_id"] == other_root, "node_id"] = self_root

        duplicate_objects_mask = self.objects["object_id"].isin(other_objects["object_id"])
        if duplicate_objects_mask.any():
            print("Objects present in both trees: {:,d}. Keeping assignments from `other`.".format(duplicate_objects_mask.sum()))
        self_objects = self.objects[~duplicate_objects_mask]

        self.nodes = pd.concat((self.nodes, other_nodes), ignore_index=True)
        self.objects = pd.concat((self_objects, other_objects), ignore_index=True)

    def to_networkx(self):
        """
        Return a NetworkX DiGraph object representing the tree.
        """
        try:
            from networkx import DiGraph
        except ImportError:
            raise ImportError('You must have networkx installed to export networkx graphs')

        result = DiGraph()
        for row in self.nodes[["parent_id", "node_id"]].values:
            result.add_edge(row[0], row[1])

        return result

    def check_connectivity(self):
        """
        Traverse the tree and see if all nodes and objects are visited.
        """


        # Visit objects
        self.nodes["__visited"] = False
        for node_idx in self.topological_order_idx():
            self.nodes.loc[node_idx, "__visited"] = True
        if any(~self.nodes["__visited"]):
            raise ValueError("Tree is not a single connected component.")
        del self.nodes["__visited"]

        # Check objects
        ons = set(self.objects["node_id"])
        nns = set(self.nodes["node_id"])
        if len(ons - nns):
            raise ValueError("Some objects are not reachable.")

    def copy(self):
        """
        Create a copy of this tree.
        """
        return Tree(self.nodes.copy(), self.objects.copy())


if __name__ == "__main__":
    sys.exit(fire.Fire(Tree))
