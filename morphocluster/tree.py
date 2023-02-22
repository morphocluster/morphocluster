"""
Created on 13.03.2018

@author: mschroeder
"""
import csv
import itertools
import os
import warnings
from numbers import Integral
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from etaprogress.progress import ProgressBar
from genericpath import commonprefix
from sklearn.cluster import KMeans
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
from sqlalchemy.sql.elements import literal_column
from sqlalchemy.sql.expression import bindparam, literal, select
from sqlalchemy.sql.functions import coalesce, func
from timer_cm import Timer
from tqdm import tqdm

from morphocluster import processing
from morphocluster.classifier import Classifier
from morphocluster.extensions import database
from morphocluster.helpers import seq2array
from morphocluster.member import MemberCollection
from morphocluster.models import (
    nodes,
    nodes_objects,
    nodes_rejected_objects,
    objects,
    projects,
)
from morphocluster.processing.prototypes import Prototypes, merge_prototypes

# TODO: Make N_PROTOTYPES configurable
N_PROTOTYPES = 16


class TreeError(Exception):
    """
    Raised by Tree if an error occurs.
    """

    pass


def _roundrobin(iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next_ in nexts:
                yield next_()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def _paths_from_common_ancestor(paths):
    """
    Strips the common prefix (without the first common ancestor) from p1 and p2.
    """
    common_anestor_idx = len(commonprefix(paths)) - 1
    return [p[common_anestor_idx:] for p in paths]


def _paths_to_node_order(paths):
    """
    TODO: Returns nodes from list of paths in bottom-up order.

    Pop last element of longest path. Do not duplicate existing nodes.
    """

    result = []

    while True:
        longest_path = max(paths, key=len)

        if not longest_path:
            break

        node = longest_path.pop()

        if node not in result:
            result.append(node)

    return result


def _rquery_preds(node_id):
    """
    Constructs a selectable of predecessor of node_id with all columns of `nodes` and an additional
    `level`.

    `level` is 0 for the supplied `node_id` and decreases for each predecessor.
    """
    # Start with the last child
    query = (
        select([nodes, literal(0).label("level")])
        .where(nodes.c.node_id == node_id)
        .cte(recursive=True)
        .alias("q")
    )

    preds = nodes.alias("p")

    query = query.union_all(
        select([preds, literal_column("level") - 1]).where(
            query.c.parent_id == preds.c.node_id
        )
    )

    return query


def _rquery_subtree(node_id, recurse_cb=None):
    """
    Constructs a selectable for the subtree rooted at node_id
    with all columns of `nodes` and an additional `level`.

    `level` is 0 for the supplied `node_id` and increases for each level of
    hierarchy.

    Parameters:
        recurse_cb: A callback with two parameters (q, s). q is the recursive query, s is the successor.
            The callback must return a clause that can be used in where().
    """
    q = (
        select([nodes, literal(0).label("level")])
        .where(nodes.c.node_id == node_id)
        .cte(recursive=True)
        .alias("q")
    )

    s = nodes.alias("s")

    rq = select([s, literal_column("level") + 1]).where(s.c.parent_id == q.c.node_id)

    if callable(recurse_cb):
        condition = recurse_cb(q, s)
        rq = rq.where(condition)

    q = q.union_all(rq)

    return q


def _compute_flags(mapping: Mapping, names: Iterable[str]):
    return {
        k: bool(mapping[k]) for k in names if k in mapping and pd.notnull(mapping[k])
    }


class Tree(object):
    """
    A tree as represented by the database.
    """

    def __init__(self, connection):
        self.connection = connection

    def load_project(self, name, tree):
        """
        Load a project from a saved tree.
        """

        if not isinstance(tree, processing.Tree):
            tree = processing.Tree.from_saved(tree)

        with self.connection.begin():
            project_id = self.create_project(name)

            # Lock project
            self.lock_project(project_id)

            progress_bar = tqdm(
                total=len(tree.nodes) + len(tree.objects), unit_scale=True
            )

            def progress_cb(nadd):
                progress_bar.update(nadd)

            for node in tree.topological_order():
                name = (
                    node["name"]
                    if "name" in node and pd.notnull(node["name"])
                    else None
                )

                object_ids = tree.objects_for_node(node["node_id"])[
                    "object_id"
                ].tolist()

                tree_parent_id = (
                    int(node["parent_id"]) if pd.notnull(node["parent_id"]) else None
                )

                flags = _compute_flags(node, ("approved", "starred", "filled"))  # type: ignore

                # Create node with objects
                self.create_node(
                    project_id,
                    orig_node_id=int(node["node_id"]),
                    orig_parent=tree_parent_id,
                    object_ids=object_ids,
                    name=name,
                    progress_cb=progress_cb,
                    **flags,
                )

                # Update progress bar
                progress_cb(1)
            progress_bar.close()
            print()

        print("Done after {}s.".format(progress_bar.format_dict["elapsed"]))

        return project_id

    def update_project(self, project_id, tree):
        """
        Update a project from a saved tree.
        """

        if not isinstance(tree, processing.Tree):
            tree = processing.Tree.from_saved(tree)

        with self.connection.begin():
            root_id = self.get_root_id(project_id)

            # Lock project
            self.lock_project(project_id)

            tree_root_id = tree.get_root_id()

            progress_bar = tqdm(total=len(tree.nodes), unit_scale=True)

            for node in tree.topological_order():
                name = (
                    node["name"]
                    if "name" in node and pd.notnull(node["name"])
                    else None
                )

                object_ids = tree.objects_for_node(node["node_id"])[
                    "object_id"
                ].tolist()

                tree_parent_id = (
                    int(node["parent_id"]) if pd.notnull(node["parent_id"]) else None
                )

                flags = _compute_flags(node, ("approved", "starred", "filled"))  # type: ignore

                # Were updating an existing project
                if tree_parent_id is None:
                    # Do not change root when updating
                    print("Skipping root.")
                else:
                    # Calculate parent
                    if tree_parent_id == tree_root_id:
                        # If tree parent is tree root, use supplied root_id
                        parent_cfg = dict(parent_id=root_id)
                    else:
                        # If node further down, use orig_parent
                        parent_cfg = dict(orig_parent=tree_parent_id)

                    # Create new node without creating new objects
                    new_node_id = self.create_node(
                        project_id,
                        orig_node_id=int(node["node_id"]),
                        name=name,
                        **parent_cfg,
                        **flags,
                    )

                    # Relocate objects (but take only from root)
                    self.relocate_objects(object_ids, new_node_id, src_node_id=root_id)

                # Update progress bar
                progress_bar.update()
            progress_bar.close()
            print()

        print("Done after {}s.".format(progress_bar.format_dict["elapsed"]))

        return project_id

    def get_orig_node_id_offset(self, project_id):
        """
        Calculate the offset for new clusters.

        This is max(orig_id) + 1
        """
        stmt = select([func.max(nodes.c.orig_id)]).where(
            nodes.c.project_id == project_id
        )
        result = self.connection.execute(stmt).scalar()

        if result is None:
            return 0
        return result + 1

    def lock_project(self, project_id):
        """
        Acquire advisory transaction lock for a project.
        """
        return self.connection.execute(select([func.pg_advisory_xact_lock(project_id)]))

    def lock_project_for_node(self, node_id):
        """
        Acquire advisory project lock given a node ID.
        """
        project_id = (
            select([nodes.c.project_id]).where(nodes.c.node_id == node_id).as_scalar()
        )
        return self.lock_project(project_id)

    def __load_project_old(self, name, path, root_first=True):
        tree_fn = os.path.join(path, "tree.csv")
        objids_fn = os.path.join(path, "objids.csv")

        raw_tree = pd.read_csv(
            tree_fn,
            index_col=False,
            dtype={
                "parent": np.uint64,
                "child": np.uint64,
                "lambda_val": np.float64,
                "child_size": np.uint64,
                "name": str,
            },
        )

        objids = pd.read_csv(
            objids_fn, index_col=False, names=["objid"], header=None, squeeze=True
        )

        raw_tree_nodes = raw_tree[raw_tree["child_size"] > 1].sort_values(
            "parent", ascending=root_first
        )
        raw_tree_objects = raw_tree[raw_tree["child_size"] == 1]

        root_orig_id = int(raw_tree_nodes["parent"].iloc[0])

        with self.connection.begin():
            project_id = self.create_project(name)

            bar = ProgressBar(len(raw_tree_objects) + len(raw_tree_nodes), max_width=40)

            def progress_cb(nadd):
                bar.numerator += nadd
                print(bar, end="\r")

            # Get object ids for the root
            object_idxs = raw_tree_objects[raw_tree_objects["parent"] == root_orig_id][
                "child"
            ]
            object_ids = objids[object_idxs]

            # Create root
            root_node_id = self.create_node(
                project_id, root_orig_id, object_ids=object_ids, progress_cb=progress_cb
            )

            for row in raw_tree_nodes.itertuples():
                # row.child is the current node
                # row.parent is its parent

                # Get object ids for the current node
                object_idxs = raw_tree_objects[raw_tree_objects["parent"] == row.child][
                    "child"
                ]
                object_ids = objids[object_idxs]

                name = (
                    row.name if hasattr(row, "name") and pd.notnull(row.name) else None
                )

                self.create_node(
                    project_id,
                    orig_node_id=row.child,
                    orig_parent=row.parent,
                    object_ids=object_ids,
                    name=name,
                    starred=name is not None,
                    progress_cb=progress_cb,
                )
                bar.numerator += 1

            print()

        print("Done after {}s.".format(bar._eta.elapsed))
        print("Created root (orig/real):", root_orig_id, root_node_id)

        return project_id

    def connect_supertree(self, root_id):
        with self.connection.begin():
            successors = _rquery_subtree(root_id)

            supersuccessor_ids = select([successors.c.node_id]).where(
                successors.c.starred == True
            )
            supersuccessor_ids = self.connection.execute(supersuccessor_ids).fetchall()
            supersuccessor_ids = [node_id for (node_id,) in supersuccessor_ids]
            supersuccessor_ids.insert(0, root_id)

            bar = ProgressBar(len(supersuccessor_ids), max_width=40)

            for node_id in supersuccessor_ids:

                def recurse_cb(q, _):
                    return q.c.starred == False

                successors = _rquery_subtree(node_id, recurse_cb)

                # ===============================================================
                # UPDATE nodes
                # SET superparent_id=node_id
                # WHERE node_id IN (SELECT node_id from successors);
                # ===============================================================
                successor_ids = select([successors.c.node_id]).where(
                    successors.c.node_id != node_id
                )
                stmt = (
                    nodes.update()
                    .values(superparent_id=node_id)
                    .where(nodes.c.node_id.in_(successor_ids))
                )
                self.connection.execute(stmt)

                bar.numerator += 1
                print(bar, end="\r")
            print()

    def get_objects_recursive(self, node_id):
        # Recursively select all descendants
        rquery = select([nodes]).where(nodes.c.node_id == node_id).cte(recursive=True)

        parents = rquery.alias("n")
        descendants = nodes.alias("nd")

        rquery = rquery.union_all(
            select([descendants]).where(descendants.c.parent_id == parents.c.node_id)
        )

        # For each node in rquery, get associated objects
        obj_query = (
            select([objects])
            .distinct()
            .select_from(rquery.join(nodes_objects).join(objects))
        )

        result = self.connection.execute(obj_query)

        return [dict(r) for r in result]

    def calculate_progress(self, node_id):
        """
        Calculate labeling progress.

        - Number of objects below approved nodes
        """

        with Timer("calculate_progress") as t:

            with self.connection.begin(), t.child("consolidate_node"):
                subtree = self.consolidate_node(
                    node_id, depth="full", return_="raw", descend_approved=False
                )
                """ subtree = _rquery_subtree(node_id)
                subtree = (select([subtree])
                        .order_by(subtree.c.level.desc()))
                subtree = pd.read_sql_query(
                    subtree, self.connection, index_col="node_id") """

            subtree["n_approved_objects"] = (
                subtree["approved"] * subtree["_n_objects_deep"]
            )
            subtree["n_filled_objects"] = subtree["filled"] * subtree["_n_objects_deep"]
            subtree["n_named_objects"] = (
                pd.notna(subtree["name"]) * subtree["_n_objects_deep"]
            )
            subtree["n_approved_nodes"] = subtree["approved"].astype(int)
            subtree["n_filled_nodes"] = subtree["filled"].astype(int)
            subtree["n_nodes"] = 1

            fields = [
                "_n_objects",
                "_n_objects_deep",
                "n_filled_objects",
                "n_approved_objects",
                "n_named_objects",
                "n_approved_nodes",
                "n_filled_nodes",
                "n_nodes",
            ]

            # Leaves
            leaves_mask = subtree["_n_children"] == 0
            leaves_result = subtree.loc[leaves_mask, fields].sum(axis=0).to_dict()
            leaves_result = {
                "leaves_{}".format(k.lstrip("_")): v for k, v in leaves_result.items()
            }

            # Compute deep values for root
            # subtree is ordered deep-first
            with t.child("deep stats"):
                for nid in subtree.index:
                    child_selector = subtree["parent_id"] == nid

                    subtree.at[nid, "n_approved_objects"] = max(
                        subtree.at[nid, "n_approved_objects"],
                        subtree.loc[child_selector, "n_approved_objects"].sum(),
                    )

                    subtree.at[nid, "n_named_objects"] = max(
                        subtree.at[nid, "n_named_objects"],
                        subtree.loc[child_selector, "n_named_objects"].sum(),
                    )

                    subtree.at[nid, "n_approved_nodes"] = (
                        subtree.at[nid, "n_approved_nodes"]
                        + subtree.loc[child_selector, "n_approved_nodes"].sum()
                    )

                    subtree.at[nid, "n_filled_nodes"] = (
                        subtree.at[nid, "n_filled_nodes"]
                        + subtree.loc[child_selector, "n_filled_nodes"].sum()
                    )

                    subtree.at[nid, "n_nodes"] = (
                        subtree.at[nid, "n_nodes"]
                        + subtree.loc[child_selector, "n_nodes"].sum()
                    )

            deep_result = subtree.loc[node_id, fields].to_dict()
            deep_result = {k.lstrip("_"): v for k, v in deep_result.items()}

            return dict(**leaves_result, **deep_result)

    def dump_tree(self, root_id):
        """
        Generate a processing.Tree from the tree below root_id.
        """
        with self.connection.begin():
            # Acquire project lock
            self.lock_project_for_node(root_id)

            # Get complete subtree with up to date cached values
            print("Consolidating cached values...")
            subtree = self.consolidate_node(root_id, depth="full", return_="raw")

            keep_columns = [
                "orig_id",
                "parent_id",
                "name",
                "starred",
                "filled",
                "approved",
                "_n_children",
                "_n_objects",
                "_n_objects_deep",
            ]
            tree_nodes = subtree[keep_columns].reset_index()

            print("Getting objects...")
            # Get subtree below root
            subtree = _rquery_subtree(root_id)

            # Get object IDs for all nodes
            node_objects = (
                select([nodes_objects.c.node_id, nodes_objects.c.object_id])
                .select_from(nodes_objects)
                .where(nodes_objects.c.node_id == subtree.c.node_id)
            )

            node_objects = pd.read_sql_query(node_objects, self.connection)

            node_rejected_objects = (
                select(
                    [
                        nodes_rejected_objects.c.node_id,
                        nodes_rejected_objects.c.object_id,
                    ]
                )
                .select_from(nodes_rejected_objects)
                .where(nodes_rejected_objects.c.node_id == subtree.c.node_id)
            )

            node_rejected_objects = pd.read_sql_query(
                node_rejected_objects, self.connection
            )

            try:
                tree = processing.Tree(tree_nodes, node_objects, node_rejected_objects)
            except ValueError:
                print(tree_nodes)
                print(node_objects)
                raise
        return tree

    def export_tree(self, root_id, tree_fn):
        """
        Export the whole tree with its objects.
        """

        tree = self.dump_tree(root_id)
        print("Writing tree...")
        tree.save(tree_fn)

    def get_root_id(self, project_id):
        """Get the root node ID of a project."""
        stmt = select(
            [nodes.c.node_id],
            (nodes.c.parent_id == None) & (nodes.c.project_id == project_id),
        )
        root_id = self.connection.execute(stmt).scalar()

        if root_id is None:
            raise TreeError("No root")

        return root_id

    def reset_grown(self, project_id):
        """Reset the filled (grown) flag to false for a certain project."""
        stmt = (
            nodes.update().where(nodes.c.project_id == project_id).values(filled=False)
        )

        self.connection.execute(stmt)

    def get_projects(self, visible_only=True):
        """
        Get projects with name
        """

        qroots = (
            select([nodes.c.project_id, nodes.c.node_id])
            .where(nodes.c.parent_id == None)
            .alias("roots")
        )
        qprojects = select([projects, qroots.c.node_id]).select_from(
            projects.join(qroots, qroots.c.project_id == projects.c.project_id)
        )

        if visible_only:
            qprojects = qprojects.where(projects.c.visible == True)

        result = self.connection.execute(qprojects).fetchall()

        return [dict(r) for r in result]

    def get_project(self, project_id):
        """
        Get a project by its ID.
        """

        stmt = text(
            """
        SELECT p.*, n.node_id
        FROM nodes AS n
        JOIN projects AS p
        ON p.project_id = n.project_id
        WHERE n.parent_id IS NULL AND p.project_id = :project_id
        """
        )

        result = self.connection.execute(stmt, project_id=project_id).fetchone()

        return dict(result)

    def get_path_ids(self, node_id):
        """
        Get the path of the node.

        Returns:
            List of `node_id`s.
        """
        stmt = text(
            """
            WITH RECURSIVE q AS
            (
                SELECT  n.*, 1 AS level
                FROM    nodes AS n
                WHERE   n.node_id = :node_id
                UNION ALL
                SELECT  p.*, level + 1
                FROM    q
                JOIN    nodes AS p
                ON      p.node_id = q.parent_id
            )
            SELECT  node_id
            FROM    q
            ORDER BY
            level DESC
        """
        )
        rows = self.connection.execute(stmt, node_id=node_id).fetchall()
        return [r for (r,) in rows]

    def create_project(self, name):
        """
        Create a project with a name and return its id.
        """
        stmt = projects.insert({"name": name})
        result = self.connection.execute(stmt)
        project_id = result.inserted_primary_key[0]

        return project_id

    def create_node(
        self,
        project_id=None,
        orig_node_id=None,
        parent_id=None,
        orig_parent=None,
        object_ids=None,
        name=None,
        progress_cb=None,
        **kwargs,
    ):
        """
        Create a node.

        Returns:
            node ID
        """
        if project_id is None and parent_id is not None:
            project_id = (
                select([nodes.c.project_id])
                .where(nodes.c.node_id == parent_id)
                .as_scalar()
            )
            # Make sure that the retrieved id is non-NULL by coalescing with -1 which will trigger an IntegrityError
            project_id = coalesce(project_id, -1)

        if parent_id is None and orig_parent is not None:
            # Subquery: Find parent by its orig_id
            parent_id = select(
                [nodes.c.node_id],
                (nodes.c.orig_id == orig_parent) & (nodes.c.project_id == project_id),
            ).as_scalar()
            # Make sure that the retrieved id is non-NULL by coalescing with -1 which will trigger an IntegrityError
            parent_id = coalesce(parent_id, -1)

        row = {
            "project_id": project_id,
            "parent_id": parent_id,
            "orig_id": orig_node_id,
            "name": name,
            **kwargs,
        }

        stmt = nodes.insert(row)

        try:
            result = self.connection.execute(stmt)
        except SQLAlchemyError as e:
            row["orig_parent"] = orig_parent
            raise TreeError("Node could not be created: {!r}".format(row)) from e

        node_id = result.inserted_primary_key[0]

        # Insert objects
        if object_ids is not None:
            object_ids = iter(object_ids)
            while True:
                chunk = itertools.islice(object_ids, 1000)

                data = [
                    dict(node_id=node_id, object_id=object_id, project_id=project_id)
                    for object_id in chunk
                ]

                if not data:
                    break

                self.connection.execute(nodes_objects.insert(), data)

                if callable(progress_cb):
                    progress_cb(len(data))
        return node_id

    def _calc_type_objects(self, children, objects_):
        """
        Calculate nine type objects for a node as
            a) a sample of nine type objects from its children, or
            b) nine of its own objects, if the node is a leaf.
        """
        if len(children) > 0:
            # Randomly subsample children
            subsample = np.random.choice(children, min(len(children), 9), replace=False)
            result = list(
                itertools.islice(
                    _roundrobin([c["_type_objects"] for c in subsample]), 9
                )
            )

            if len(result) == 0:
                print("\n", subsample)

            return result
        else:
            return [o["object_id"] for o in objects_[:9]]

    def _calc_own_type_objects(self, children, objects_):
        """
        Calculate nine own type objects_ for a node as
            a) the nine objects_ with maximum distance to the children, or
            b) [], if the node is a leaf.
        """

        if len(children) > 0 and len(objects_) > 0:
            try:
                classifier = Classifier(children.vectors)
                distances = classifier.distances(objects_.vectors)
                max_dist = np.max(distances, axis=0)
                max_dist_idx = np.argsort(max_dist)[::-1]

                assert len(max_dist_idx) == len(objects_), "{} != {}".format(
                    len(max_dist_idx), len(objects_)
                )

                return [objects_[i]["object_id"] for i in max_dist_idx[:9]]

            except:
                print("child_vectors", children.vectors.shape)
                print("object_vectors", objects_.vectors.shape)
                raise

        else:
            return []

    def _calc_n_objects_deep(self, node, children):
        """
        Recursively calculate the number of objects.
        """

        child_ns = [c["_n_objects_deep"] for c in children]

        if any(n is None for n in child_ns):
            return None

        return int(node["_n_objects"] + sum(child_ns))

    def _query_n_objects_deep(self, node):
        # Recursively select all descendants
        rquery = (
            select([nodes])
            .where(nodes.c.node_id == node["node_id"])
            .cte(recursive=True)
        )

        parents = rquery.alias("n")
        descendants = nodes.alias("nd")

        rquery = rquery.union_all(
            select([descendants]).where(descendants.c.parent_id == parents.c.node_id)
        )

        # For each node in rquery, calculate #objects
        deep_count = (
            select(
                [rquery.c.node_id, func.count(nodes_objects.c.object_id).label("count")]
            )
            .select_from(rquery.join(nodes_objects))
            .group_by(rquery.c.node_id)
            .alias("deep_count")
        )

        # Build total sum
        stmt = select([func.sum(deep_count.c.count)]).select_from(deep_count)

        result = self.connection.scalar(stmt) or 0

        return int(result)

    def node_n_descendants(self, node_id):
        # Recursively select all descendants
        rquery = select([nodes]).where(nodes.c.node_id == node_id).cte(recursive=True)

        parents = rquery.alias("n")
        descendants = nodes.alias("nd")

        rquery = rquery.union_all(
            select([descendants]).where(descendants.c.parent_id == parents.c.node_id)
        )

        # Count results
        stmt = select([func.count()]).select_from(rquery)

        result = self.connection.scalar(stmt) or 0

        return result

    def get_node(self, node_id, require_valid=True):
        assert isinstance(node_id, Integral), "node_id is not integral: {!r}".format(
            node_id
        )

        if require_valid:
            # TODO: Directly use values instead of reading again from DB
            self.consolidate_node(node_id)

        stmt = select([nodes]).where(nodes.c.node_id == node_id)

        result = self.connection.execute(stmt, node_id=node_id).fetchone()

        if result is None:
            raise TreeError("Node {} is unknown.".format(node_id))

        return dict(result)

    def get_children(
        self, node_id, require_valid=True, order_by=None, include=None, supertree=False
    ):
        """
        Parameters:
            node_id: node_id of the parent node.
            require_valid (bool): Are valid cache values required?
            include ("starred" | "unstarred" | None):
                None: return all chilren.
                "starred": Return only starred children.
                "unstarred": Return only unstarred children.

        Returns:
            A list children of node_id: [{"node_id": ..., }, ...]

        """

        assert isinstance(node_id, Integral), "node_id is not integral: {!r}".format(
            node_id
        )

        if require_valid:
            self.consolidate_node(node_id, depth="children")

        stmt = select([nodes])

        if supertree:
            stmt = stmt.where(nodes.c.superparent_id == node_id)
        else:
            stmt = stmt.where(nodes.c.parent_id == node_id)

        if include is not None:
            stmt = stmt.where(nodes.c.starred == (include == "starred"))

        if order_by is not None:
            stmt = stmt.order_by(text(order_by))

        result = self.connection.execute(stmt, node_id=node_id).fetchall()

        return [dict(r) for r in result]

    def merge_node_into(self, node_id, dest_node_id):
        """
        Merge a node n into another node d.

        All objects of n will be assigned to d.
        All children of n will be assigned to d.
        n will be deleted.
        """

        with self.connection.begin():
            # Change node for objects
            stmt = (
                nodes_objects.update()
                .values(node_id=dest_node_id)
                .where(nodes_objects.c.node_id == node_id)
            )
            self.connection.execute(stmt)

            # Change parent for children
            stmt = (
                nodes.update()
                .values(parent_id=dest_node_id)
                .where(nodes.c.parent_id == node_id)
            )
            self.connection.execute(stmt)

            # Delete node
            stmt = nodes.delete(nodes.c.node_id == node_id)
            self.connection.execute(stmt)

            # Invalidate dest node
            stmt = (
                nodes.update()
                .values(cache_valid=False)
                .where(nodes.c.node_id == dest_node_id)
            )
            self.connection.execute(stmt)

            # TODO: Unapprove

    def get_objects(self, node_id, offset=None, limit=None, order_by=None):
        """
        Get objects directly below a node.
        """
        stmt = (
            select([objects])
            .select_from(objects.join(nodes_objects))
            .where(nodes_objects.c.node_id == node_id)
        )

        if order_by is not None:
            stmt = stmt.order_by(order_by)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = self.connection.execute(stmt, node_id=node_id).fetchall()

        return [dict(r) for r in result]

    def get_n_objects(self, node_id):
        stmt = (
            select([func.count()])
            .select_from(nodes_objects)
            .where(nodes_objects.c.node_id == node_id)
        )

        return self.connection.execute(stmt, node_id=node_id).scalar()

    # TODO: Also remove approval for automatically classified members
    def invalidate_node_and_parents(self, node_id):
        """
        Invalidate the cached values in the node and its parents.
        """

        with self.connection.begin():
            self.lock_project_for_node(node_id)

            stmt = text(
                """
            WITH RECURSIVE q AS
            (
                SELECT  n.*, 1 AS level
                FROM    nodes AS n
                WHERE   n.node_id = :node_id
                UNION ALL
                SELECT  p.*, level + 1
                FROM    q
                JOIN    nodes AS p
                ON      p.node_id = q.parent_id
            )
            UPDATE nodes
            SET cache_valid = FALSE
            WHERE node_id IN (SELECT node_id from q);
            """
            )

            self.connection.execute(stmt, node_id=node_id)

    def recommend_children(self, node_id, max_n=1000):
        node = self.get_node(node_id)

        # Get the path to the node
        path = self.get_path_ids(node_id)

        nodes_ = []

        # Traverse the path in reverse (without the node itself)
        for parent_id in path[:-1][::-1]:
            # TODO: limit number of children like in recommend_objects
            nodes_.extend(
                c for c in self.get_children(parent_id) if c["node_id"] not in path
            )

            # Break if we already have enough nodes
            if len(nodes_) > max_n:
                break

        # TODO: Prototypes
        vectors = [n["_centroid"] for n in nodes_]

        nodes_ = np.array(nodes_, dtype=object)
        vectors = seq2array(vectors, len(vectors))

        distances = np.linalg.norm(vectors - node["_centroid"], axis=1)

        assert len(distances) == len(vectors), distances.shape

        order = np.argsort(distances)[:max_n]

        return nodes_[order].tolist()

    def recommend_objects(self, node_id, max_n=1000):
        """
        Recommend objects for a node.

        Note: Most time is spent querying all objects below a node.
        This can be sped up, if the nodes a relatively small.

        History:
            18/10/29: Query all objects, then sort and truncate.
            pre 18/10/29: Queried number of objects per node is limited by max_n.
                This leads to suboptimal results, as the closest objects may not
                lie under these max_n quasi-randomly chosen objects.
        """
        with Timer("Tree.recommend_objects") as timer:
            node = self.get_node(node_id)
            project_id = node["project_id"]

            # Get the path to the node (without the node itself)
            path = self.get_path_ids(node_id)[:-1]

            objects_ = []

            rejected_object_ids = select([nodes_rejected_objects.c.object_id]).where(
                nodes_rejected_objects.c.node_id == node_id
            )

            prots: Prototypes = node["_prototypes"]
            if prots is None:
                raise TreeError(f"Node {node_id} has no prototypes!")

            distances_expression = [
                objects.c.vector.dist_euclidean(p) for p in prots.prototypes_
            ]

            print("Tree.recommend_objects: Querying candidates...")
            with timer.child("Query matching objects") as c:
                # Traverse the parse in reverse
                for parent_id in path[::-1]:
                    n_left = max_n - len(objects_)

                    # Break if we already have enough objects
                    if n_left <= 0:
                        break

                    # Get objects below parent_id that are not rejected by node_id
                    stmt = (
                        select(
                            [
                                objects.c.object_id,
                                objects.c.path,
                                func.least(*distances_expression).label("distance"),
                            ]
                        )
                        .select_from(objects.join(nodes_objects))
                        .where(
                            (nodes_objects.c.node_id == parent_id)
                            & (nodes_objects.c.project_id == project_id)
                            & (~objects.c.object_id.in_(rejected_object_ids))
                        )
                    )

                    with c.child("execute"):
                        r = self.connection.execute(stmt)

                    with c.child("fetch"):
                        result = r.fetchall()

                    objects_.extend(dict(r) for r in result)

            if not objects_:
                return []

            with timer.child("Convert to array"):
                distances = np.array([o["distance"] for o in objects_])
                objects_ = np.array(objects_, dtype=object)

            with timer.child("Sorting"):
                order = np.argsort(distances)[:max_n]

            with timer.child("Result assembly"):
                return objects_[order].tolist()

    def invalidate_nodes(self, nodes_to_invalidate, unapprove=False):
        """
        Invalidate the provided nodes.

        Parameters:
            nodes_to_invalidate: Collection of `node_id`s
        """

        # TODO: Get project ids for requested node ids and lock the projects
        # Otherwise a deadlock might occur.

        values = {nodes.c.cache_valid: False}

        if unapprove:
            values[nodes.c.approved] = False

        stmt = (
            nodes.update()
            .values(values)
            .where(nodes.c.node_id.in_(nodes_to_invalidate))
        )
        self.connection.execute(stmt)

    def relocate_nodes(self, node_ids, parent_id, unapprove=False):
        """
        Relocate nodes to another parent.
        """

        if len(node_ids) == 0:
            return

        with self.connection.begin():
            # Acquire project lock
            self.lock_project_for_node(parent_id)

            new_parent_path = self.get_path_ids(parent_id)

            # Check if the new parent is below the node
            new_parent_set = set(new_parent_path)
            for node_id in node_ids:
                if node_id in new_parent_set:
                    raise TreeError(
                        "Relocating {} to {} would create a circle!".format(
                            node_id, parent_id
                        )
                    )

            # stmt = nodes.update().values({"parent_id": parent_id}).where(nodes.c.node_id.in_(node_ids))

            # Return distinct old `parent_id`s
            stmt = text(
                """
            WITH updater AS (
                UPDATE nodes x
                SET parent_id = :new_parent_id
                FROM  (SELECT node_id, parent_id FROM nodes WHERE node_id IN :node_ids FOR UPDATE) y
                WHERE  x.node_id = y.node_id
                RETURNING y.parent_id AS old_parent_id
            )
            SELECT DISTINCT old_parent_id FROM updater;
            """
            )

            # Fetch old_parent_ids
            result = self.connection.execute(
                stmt, new_parent_id=parent_id, node_ids=tuple(node_ids)
            ).fetchall()

            # Invalidate subtree rooted at first common ancestor
            parent_paths = [new_parent_path] + [
                self.get_path_ids(r["old_parent_id"]) for r in result
            ]
            paths_to_update = _paths_from_common_ancestor(parent_paths)
            nodes_to_invalidate = set(sum(paths_to_update, []))

            assert parent_id in nodes_to_invalidate

            self.invalidate_nodes(nodes_to_invalidate, unapprove)

    def relocate_objects(self, object_ids, node_id, unapprove=False, src_node_id=None):
        """
        Relocate an object to another node.

        Args:
            src_node_id: If not None, transfer only objects from this node.

        TODO: This is slow!
        """

        if len(object_ids) == 0:
            return

        with self.connection.begin():
            # Acquire project lock
            self.lock_project_for_node(node_id)

            # Poject id of the new node
            project_id = select([nodes.c.project_id]).where(nodes.c.node_id == node_id)
            project_id = self.connection.execute(project_id).scalar()

            new_node_path = self.get_path_ids(node_id)

            # Find current node_ids of the objects
            # This is slow!
            # old_node_ids is required for invalidation
            # TODO: Distinct!
            stmt = (
                select([nodes_objects.c.node_id])
                .with_for_update()
                .where(
                    nodes_objects.c.object_id.in_(object_ids)
                    & (nodes_objects.c.project_id == project_id)
                )
            )

            if src_node_id is not None:
                stmt = stmt.where(nodes_objects.c.node_id == src_node_id)

            old_node_ids = [
                r["node_id"] for r in self.connection.execute(stmt).fetchall()
            ]

            # Update assignments
            stmt = (
                nodes_objects.update()
                .values({"node_id": node_id})
                .where(
                    nodes_objects.c.object_id.in_(object_ids)
                    & (nodes_objects.c.project_id == project_id)
                )
            )

            if src_node_id is not None:
                stmt = stmt.where(nodes_objects.c.node_id == src_node_id)

            self.connection.execute(stmt)

            # # Return distinct old `parent_id`s
            # stmt = text("""
            # WITH updater AS (
            #     UPDATE nodes_objects x
            #     SET node_id = :node_id
            #     FROM  (SELECT object_id, node_id FROM nodes_objects
            #         WHERE project_id = :project_id AND object_id IN :object_ids FOR UPDATE) y
            #     WHERE  project_id = :project_id AND x.object_id = y.object_id
            #     RETURNING y.node_id AS old_node_id
            # )
            # SELECT DISTINCT old_node_id FROM updater;
            # """)

            # # Fetch old_parent_ids
            # result = self.connection.execute(stmt,
            #                                  node_id=node_id,
            #                                  project_id=project_id,
            #                                  object_ids=tuple(object_ids)
            #                                  ).fetchall()

            # Invalidate subtree rooted at first common ancestor
            paths = [new_node_path] + [old_node_ids]
            paths_to_update = _paths_from_common_ancestor(paths)
            nodes_to_invalidate = set(sum(paths_to_update, []))

            print("Invalidating {!r}...".format(nodes_to_invalidate))

            assert node_id in nodes_to_invalidate

            self.invalidate_nodes(nodes_to_invalidate, unapprove)

    def reject_objects(self, node_id, object_ids):
        """
        Save objects as rejected for a certain node_id to prevent further recommendation.
        """

        if not object_ids:
            return

        with self.connection.begin():
            # Acquire project lock
            self.lock_project_for_node(node_id)

            # Poject id of the node
            project_id = select([nodes.c.project_id]).where(nodes.c.node_id == node_id)
            project_id = self.connection.execute(project_id).scalar()

            new_node_path = self.get_path_ids(node_id)

            # Update assignments
            stmt = nodes_rejected_objects.insert()
            self.connection.execute(
                stmt,
                [
                    {"node_id": node_id, "project_id": project_id, "object_id": oid}
                    for oid in object_ids
                ],
            )

    def update_node(self, node_id, data):
        if "parent_id" in data:
            warnings.warn("parent_id in data")

        if data.pop("node_id", None) is not None:
            raise TreeError("Do not update the node_id!")

        stmt = nodes.update().values(data).where(nodes.c.node_id == node_id)
        self.connection.execute(stmt)

    def get_tip(self, node_id):
        """
        Get the id of the tip (descendant with maximum depth) below a node.

        A node is selected as tip if
            - it is not approved
            - it is not starred
            - is has children
        """

        stmt = text(
            """
        WITH    RECURSIVE
        q AS
        (
            SELECT  node_id, starred, 1 as level
            FROM    nodes AS n
            WHERE   node_id = :node_id
            UNION ALL
            SELECT  nd.node_id, nd.starred, level + 1
            FROM    q
            JOIN    nodes AS nd
            ON      nd.parent_id = q.node_id
            WHERE nd.approved = 'f' AND nd.starred = 'f'
        )
        SELECT q.node_id
        FROM q LEFT JOIN nodes as c ON c.parent_id = q.node_id
        GROUP BY q.node_id, q.level
        HAVING COUNT(c.*) > 0
        ORDER BY q.level desc
        LIMIT 1
        """
        )

        return self.connection.execute(stmt, node_id=node_id).scalar()

    def get_minlevel_starred(self, root_node_id, require_valid=True):
        """
        Returns all starred nodes with minimal depth.

        Descend into the tree while a node is not starred. Return only starred nodes.
        """

        rquery = (
            select([nodes]).where(nodes.c.node_id == root_node_id).cte(recursive=True)
        )

        parents = rquery.alias("n")
        descendants = nodes.alias("nd")

        rquery = rquery.union_all(
            # Include descendants when the parent is not starred
            select([descendants]).where(
                (not parents.c.starred) & (descendants.c.parent_id == parents.c.node_id)
            )
        )

        stmt = select([rquery]).where(rquery.c.starred)

        result = self.connection.execute(stmt).fetchall()

        return [
            self._upgrade_node(dict(r), require_valid=require_valid) for r in result
        ]

    def get_next_node(
        self,
        node_id,
        leaf=False,
        recurse_cb=None,
        filter=None,
        preferred_first=False,
        order_by=None,
    ):
        """
        Get the id of the next unapproved node.

        This is either
            a) the deepest node below, if this current node matches the recurse_cb, or
            b) the first node below a predecessor of the current node that does not match the recurse_cb.

        Parameters:
            node_id
            leaf: Only return leaves.
        """

        # First try if there are candidates below this node
        subtree = _rquery_subtree(node_id, recurse_cb)

        # TODO: Could be replaced by cached values
        children = nodes.alias("children")
        n_children = (
            select([func.count()])
            .select_from(children)
            .where(children.c.parent_id == subtree.c.node_id)
            .as_scalar()
            .label("n_children")
        )

        n_objects = (
            select([func.count()])
            .select_from(nodes_objects)
            .where(nodes_objects.c.node_id == subtree.c.node_id)
            .as_scalar()
            .label("n_objects")
        )

        stmt = select([subtree.c.node_id])

        if filter is not None:
            stmt = stmt.where(filter(subtree))

        if leaf:
            stmt = stmt.where(n_children == 0)

        if preferred_first:
            stmt = stmt.order_by(subtree.c.preferred.desc())

        stmt = stmt.order_by(subtree.c.level.desc()).limit(1)

        if order_by is None:
            pass
        elif order_by == "largest":
            stmt = stmt.order_by(n_objects.desc())
        elif order_by == "smallest":
            stmt = stmt.order_by(n_objects.asc())
        else:
            raise ValueError(f"Unknown order_by value: {order_by}")

        print(stmt)

        result = self.connection.execute(stmt).scalar()

        if result is not None:
            return result

        # Otherwise go to parent
        node = self.connection.execute(nodes.select().where(nodes.c.node_id == node_id)).first()

        if node["parent_id"]:
            print("No matching children, trying parent: {}".format(node["parent_id"]))
            return self.get_next_node(node["parent_id"], leaf, recurse_cb, filter)

        return None

    def consolidate_node(self, node_id, depth=0, descend_approved=True, return_=None):
        """
        Ensures that the calculated values of this node are valid.

        If deep=True, ensures that also the calculated values of all successors are valid.

        Parameters:
            node_id: Root of the subtree that gets consolidated.
            depth: Ensure validity of cached values at least up to a certain depth.
            return_: None | "node" | "children". Return this node or its children.

        Returns:
            node dict or list of children, depending on return_ parameter.
        """

        with Timer("consolidate_node") as t:

            if isinstance(depth, str):
                if depth == "children":
                    depth = 1
                elif depth == "full":
                    depth = -1
                else:
                    raise NotImplementedError("Unknown depth string: {}".format(depth))

            # Wrap everything in a transaction
            with self.connection.begin():
                # Acquire project lock
                self.lock_project_for_node(node_id)

                if depth == -1:
                    if descend_approved:
                        recurse_cb = None
                    else:
                        # Only recurse into invalid nodes
                        # Ensure validity up to a certain level
                        def recurse_cb(q, s):
                            return (q.c.cache_valid == False) | (q.c.approved == False)

                else:
                    if not descend_approved:
                        raise NotImplementedError()

                    # Only recurse into invalid nodes
                    # Ensure validity up to a certain level
                    def recurse_cb(q, s):
                        return (q.c.cache_valid == False) | (q.c.level < depth)

                invalid_subtree = _rquery_subtree(node_id, recurse_cb)

                # Readily query real n_objects
                n_objects = (
                    select([func.count()])
                    .select_from(nodes_objects)
                    .where(nodes_objects.c.node_id == invalid_subtree.c.node_id)
                    .as_scalar()
                    .label("_n_objects_")
                )

                # Readily query real n_children
                children = nodes.alias("children")
                n_children = (
                    select([func.count()])
                    .select_from(children)
                    .where(children.c.parent_id == invalid_subtree.c.node_id)
                    .as_scalar()
                    .label("_n_children_")
                )

                stmt = select([invalid_subtree, n_objects, n_children]).order_by(
                    invalid_subtree.c.level.desc()
                )

                with t.child("read_sql_query"):
                    invalid_subtree = pd.read_sql_query(
                        stmt, self.connection, index_col="node_id"
                    )

                if len(invalid_subtree) == 0:
                    raise TreeError("Unknown node: {}".format(node_id))

                if not invalid_subtree["cache_valid"].all():
                    # 1. _n_objects, _n_children
                    invalid_subtree["_n_objects"] = invalid_subtree["_n_objects_"]
                    invalid_subtree["_n_children"] = invalid_subtree["_n_children_"]

                    invalid_subtree["__updated"] = False

                    # Initialize clusterer
                    clusterer = KMeans(N_PROTOTYPES, n_init=2)

                    # Iterate over DataFrame fixing the values along the way
                    bar = ProgressBar(len(invalid_subtree), max_width=40)
                    for node_id in invalid_subtree.index:
                        try:
                            if invalid_subtree.at[node_id, "cache_valid"]:
                                # Don't recalculate valid nodes as invalid_subtree (rightly)
                                # doesn't include their children.
                                continue

                            child_selector = invalid_subtree["parent_id"] == node_id
                            children = invalid_subtree.loc[child_selector]
                            # Build collection of children. (Set centroid of children without a vector to zero to allow alignment with cardinalities.)
                            children_dict = MemberCollection(
                                children.reset_index().to_dict("records"), "zero"
                            )

                            # 2. _n_objects_deep
                            _n_objects = invalid_subtree.loc[node_id, "_n_objects"]
                            _n_objects_deep = (
                                _n_objects + children["_n_objects_deep"].sum()
                            )
                            invalid_subtree.at[
                                node_id, "_n_objects_deep"
                            ] = _n_objects_deep

                            # Sample 1000 objects to speed up the calculation
                            with t.child("get_objects"):
                                objects_ = MemberCollection(
                                    self.get_objects(
                                        node_id, order_by=objects.c.rand, limit=1000
                                    ),
                                    "raise",
                                )

                            # 3. _own_type_objects, _type_objects
                            # TODO: Replace _own_type_objects with "_atypical_objects"
                            with t.child("_calc_own_type_objects"):
                                invalid_subtree.at[
                                    node_id, "_own_type_objects"
                                ] = self._calc_own_type_objects(children_dict, objects_)
                                # self._calc_own_type_objects(children_dict, objects_)

                            with t.child("_calc_type_objects"):
                                invalid_subtree.at[
                                    node_id, "_type_objects"
                                ] = self._calc_type_objects(children_dict, objects_)

                            if (
                                len(children_dict) > 0
                                and len(invalid_subtree.at[node_id, "_type_objects"])
                                == 0
                            ):
                                print(
                                    "\nNode {} has no type objects although it has children!".format(
                                        node_id
                                    )
                                )

                            # 4. _centroid
                            with t.child("_centroid"):
                                _centroid = []
                                _centroid_support = 0

                                if len(objects_) > 0:
                                    # Object mean, weighted with number of objects
                                    _centroid.append(np.sum(objects_.vectors, axis=0))
                                    _centroid_support += len(objects_)

                                if len(children_dict) > 0:
                                    # Children mean
                                    cardinalities = children_dict.cardinalities
                                    children_mean = np.sum(
                                        cardinalities[:, np.newaxis]
                                        * children_dict.vectors,
                                        axis=0,
                                    )
                                    _centroid.append(children_mean)
                                    _centroid_support += cardinalities.sum()

                                if len(_centroid) > 0 and _centroid_support > 0:
                                    _centroid = (
                                        np.sum(_centroid, axis=0) / _centroid_support
                                    )
                                else:
                                    _centroid = None

                                invalid_subtree.at[node_id, "_centroid"] = _centroid

                                if invalid_subtree.loc[node_id, "_centroid"] is None:
                                    print("\nNode {} has no centroid!".format(node_id))

                            # 5. _prototypes
                            with t.child("_prototypes"):
                                _prototypes = []

                                if len(objects_) > 0:
                                    prots = Prototypes(clusterer)
                                    prots.fit(objects_.vectors)
                                    _prototypes.append(prots)
                                if len(children_dict) > 0:
                                    _prototypes.extend(
                                        c["_prototypes"]
                                        for c in children_dict
                                        if c["_prototypes"] is not None
                                    )

                                if len(_prototypes) > 0:
                                    try:
                                        _prototypes = merge_prototypes(
                                            _prototypes, N_PROTOTYPES
                                        )
                                    except:
                                        for prots in _prototypes:
                                            print(prots.prototypes_)
                                        raise
                                else:
                                    _prototypes = None
                                    print(
                                        "\nNode {} has no prototypes!".format(node_id)
                                    )

                                invalid_subtree.at[node_id, "_prototypes"] = _prototypes

                            # Finally, flag as updated
                            invalid_subtree.at[node_id, "__updated"] = True

                            bar.numerator += 1
                            print(node_id, bar, end="    \r")
                        except:
                            print(f"Error processing node {node_id}")
                            raise
                    print()

                    # Convert _n_objects_deep to int (might be object when containing NULL values in the database)
                    invalid_subtree["_n_objects_deep"] = invalid_subtree[
                        "_n_objects_deep"
                    ].astype(int)

                    # Flag all rows as valid
                    invalid_subtree["cache_valid"] = True

                    # Mask for updated rows
                    updated_selection = invalid_subtree["__updated"] == True
                    n_updated = updated_selection.sum()

                    # Write back to database (if necessary)
                    if n_updated > 0:
                        # Write results back to database
                        update_fields = [
                            "cache_valid",
                            "_centroid",
                            "_prototypes",
                            "_type_objects",
                            "_own_type_objects",
                            "_n_objects_deep",
                            "_n_objects",
                            "_n_children",
                        ]

                        stmt = (
                            nodes.update()
                            .where(nodes.c.node_id == bindparam("_node_id"))
                            .values({k: bindparam(k) for k in update_fields})
                        )

                        # Build the result list of dicts with _node_id and only update_fields
                        result = invalid_subtree.loc[updated_selection, update_fields]
                        result.index.rename("_node_id", inplace=True)
                        result.reset_index(inplace=True)

                        self.connection.execute(stmt, result.to_dict("records"))

                        print("Updated {:d} nodes.".format(n_updated))

                if return_ == "node":
                    return invalid_subtree.loc[node_id].to_dict()

                if return_ == "children":
                    return invalid_subtree[
                        invalid_subtree["parent_id"] == node_id
                    ].to_dict("records")

                if return_ == "raw":
                    return invalid_subtree


if __name__ in ("__main__", "builtins"):

    def main():
        #: :type conn: sqlalchemy.engine.base.Connection
        with database.engine.connect() as conn:
            # =======================================================================
            # conn.execute("DROP TABLE IF EXISTS nodes_objects, nodes, projects;")
            # database.metadata.create_all(conn)
            # =======================================================================

            tree = Tree(conn)

            project_path = "/data1/mschroeder/NoveltyDetection/Results/CV-Clustering/2018-02-08-12-55-06/min_cluster_size-20_split-0"
            project_name = os.path.basename(os.path.normpath(path))
            print("Loading...")
            project_id = tree.load_project(project_name, project_path)
            root_id = tree.get_root_id(project_id)
            print(project_id, root_id)
            print("Simplifying...")
            tree.flatten_tree(root_id)
            tree.prune_chains(root_id)

            # ===========================================================================
            #     root_id = tree.get_root_id(1)
            #
            #     print("Root", root_id)
            #
            #     print("Children of root (before flattening):", tree.get_children(root_id))
            #
            #     tree.flatten_tree(root_id)
            #
            #     print("Children of root (after flattening):", tree.get_children(root_id))
            #
            #     tree.prune_chains(root_id)
            #
            #     print("Children of root (after pruning):", tree.get_children(root_id))
            # ===========================================================================

            print(tree.get_projects())

    main()
