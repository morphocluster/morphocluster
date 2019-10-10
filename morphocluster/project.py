import itertools
import typing as T
from genericpath import commonprefix

import numpy as np
import pandas as pd
import tqdm
from sqlalchemy import func, select
from sqlalchemy.engine import Transaction
from sqlalchemy.sql import text
from sqlalchemy.sql.elements import literal_column
from sqlalchemy.sql.expression import literal, select
from sqlalchemy.sql.functions import func
from timer_cm import Timer

from morphocluster import processing
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


class ProjectError(Exception):
    pass


def _paths_from_common_ancestor(paths):
    """
    Strip the common prefix (without the first common ancestor) from p1 and p2.
    """
    common_anestor_idx = len(commonprefix(paths)) - 1
    return [p[common_anestor_idx:] for p in paths]


def _rquery_subtree(node_id, recurse_cb=None):
    """
    Construct a selectable for the subtree rooted at node_id.

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


class Project:
    """
    Abstract a project in the database.
    """

    @staticmethod
    def create(name, dataset_id) -> "Project":
        """
        Create a project with a name.

        Returns:
            Project instance
        """
        connection = database.get_connection()

        with connection.begin():
            stmt = projects.insert({"name": name, "dataset_id": dataset_id})

            result = connection.execute(stmt)
            project_id = result.inserted_primary_key[0]

            # Create partition for nodes_objects
            stmt = "CREATE TABLE nodes_objects_{project_id} PARTITION OF nodes_objects FOR VALUES IN ({project_id})".format(
                project_id=project_id
            )
            result = connection.execute(stmt)

            # Create partition for nodes_rejected_objects
            stmt = "CREATE TABLE nodes_rejected_objects_{project_id} PARTITION OF nodes_rejected_objects FOR VALUES IN ({project_id})".format(
                project_id=project_id
            )
            result = connection.execute(stmt)

        return Project(project_id)

    @staticmethod
    def get_all(visible_only=True):
        """Get a list of all projects.
        """

        qroots = (
            select([nodes.c.project_id, nodes.c.node_id])
            .where(nodes.c.parent_id == None)
            .alias("roots")
        )
        qprojects = select([projects, qroots.c.node_id]).select_from(
            projects.join(
                qroots, qroots.c.project_id == projects.c.project_id, isouter=True
            )
        )

        if visible_only:
            qprojects = qprojects.where(projects.c.visible == True)

        connection = database.get_connection()
        result = connection.execute(qprojects).fetchall()

        return [dict(r) for r in result]

    def __init__(self, project_id):
        self.project_id = project_id

        self._connection = database.get_connection()
        self._transactions: T.List[Transaction] = []
        self._locked = 0

    def import_tree(self, tree):
        """Fill the project with the supplied tree.

        Returns:
            A Project instance
        """

        assert self._locked

        print("Loading {}...".format(tree))

        # Bail if project already contains a tree
        stmt = (
            select([func.count()])
            .select_from(nodes)
            .where(nodes.c.project_id == self.project_id)
        )
        if self._connection.execute(stmt).scalar() != 0:
            raise RuntimeError(
                "Project project_id={} already contains a tree.".format(self.project_id)
            )

        if not isinstance(tree, processing.Tree):
            tree = processing.Tree.from_saved(tree)

        pbar = tqdm.tqdm(total=len(tree.nodes) + len(tree.objects))

        for node in tree.topological_order():
            name = node["name"] if "name" in node and pd.notnull(node["name"]) else None

            object_ids = tree.objects_for_node(node["node_id"])["object_id"].tolist()
            parent_id = (
                int(node["parent_id"]) if pd.notnull(node["parent_id"]) else None
            )

            flag_names = ("approved", "starred", "filled")
            flags = {
                k: bool(node[k])
                for k in flag_names
                if k in node and pd.notnull(node[k])
            }

            self.create_node(
                node_id=int(node["node_id"]),
                parent_id=parent_id,
                object_ids=object_ids,
                name=name,
                progress_cb=pbar.update,
                **flags,
            )

            # Update progress bar
            pbar.update(1)
        pbar.close()

    def create_node(
        self,
        node_id=None,
        parent_id=None,
        object_ids=None,
        name=None,
        progress_cb=None,
        **kwargs,
    ):
        """
        Create a node.

        Returns: node ID
        """

        assert self._locked

        if node_id is None:
            stmt = select([func.max(nodes.c.node_id)]).where(
                nodes.c.project_id == self.project_id
            )
            node_id = self._connection.execute(stmt).scalar() + 1

        # Insert node
        stmt = nodes.insert(
            {
                "project_id": self.project_id,
                "parent_id": parent_id,
                "node_id": node_id,
                "name": name,
                **kwargs,
            }
        )

        self._connection.execute(stmt)

        # Insert object_ids
        if object_ids is not None:
            object_ids = iter(object_ids)
            while True:
                chunk = itertools.islice(object_ids, 1000)

                data = [
                    dict(
                        project_id=self.project_id, node_id=node_id, object_id=object_id
                    )
                    for object_id in chunk
                ]

                if not data:
                    break

                # pylint: disable=no-value-for-parameter
                self._connection.execute(nodes_objects.insert(), data)

                if callable(progress_cb):
                    progress_cb(len(data))
        return node_id

    def export_tree(self, tree_fn=None, root_id=None) -> T.Optional[processing.Tree]:
        """
        Export the tree in the database as a processing.Tree.
        Optionally, save the tree to a file.
        """

        assert self._locked

        if root_id is None:
            root_id = self.get_root_id()

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

        node_objects = pd.read_sql_query(node_objects, self._connection)

        node_rejected_objects = (
            select(
                [nodes_rejected_objects.c.node_id, nodes_rejected_objects.c.object_id]
            )
            .select_from(nodes_rejected_objects)
            .where(nodes_rejected_objects.c.node_id == subtree.c.node_id)
        )

        node_rejected_objects = pd.read_sql_query(
            node_rejected_objects, self._connection
        )

        try:
            tree = processing.Tree(tree_nodes, node_objects, node_rejected_objects)
        except ValueError:
            print(tree_nodes)
            print(node_objects)
            raise

        if tree_fn is not None:
            tree.save(tree_fn)
            return

        return tree

    def get_root_id(self) -> int:
        """Get node_id of the project's root.

        Returns:
            node_id of the project's root or None.
        """

        assert self._locked

        stmt = select(
            [nodes.c.node_id],
            (nodes.c.parent_id == None) & (nodes.c.project_id == self.project_id),
        )
        root_id = self._connection.execute(stmt).scalar()

        if root_id is None:
            raise ValueError("Project has no root.")

        return root_id

    def to_dict(self) -> T.Dict:
        """Get the properties of the project as a dict.
        """

        assert self._locked

        stmt = text(
            """
        SELECT p.*, n.node_id
        FROM nodes AS n
        JOIN projects AS p
        ON p.project_id = n.project_id
        WHERE n.parent_id IS NULL AND p.project_id = :project_id
        """
        )

        result = self._connection.execute(stmt, project_id=self.project_id).fetchone()

        return dict(result)

    def relocate_objects(self, object_ids, node_id, unapprove=False):
        """
        Relocate an object to another node.

        TODO: This is slow!
        """

        if not object_ids:
            return

        assert self._locked

        new_node_path = self.get_path(node_id)

        # Find current node_ids of the objects
        # This is slow!
        # old_node_ids is required for invalidation
        # TODO: Distinct!
        stmt = select([nodes_objects.c.node_id], for_update=True).where(
            nodes_objects.c.object_id.in_(object_ids)
            & (nodes_objects.c.project_id == self.project_id)
        )
        result = self._connection.execute(stmt)
        old_node_ids = [r["node_id"] for r in result.fetchall()]

        # Update assignments
        stmt = (
            # pylint: disable=no-value-for-parameter
            nodes_objects.update()
            .values({"node_id": node_id})
            .where(
                nodes_objects.c.object_id.in_(object_ids)
                & (nodes_objects.c.project_id == self.project_id)
            )
        )
        self._connection.execute(stmt)

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

    def relocate_nodes(self, node_ids, parent_id, unapprove=False):
        """
        Relocate nodes to another parent.

        Args:
            unapprove (bool): Unapprove the new parent.
        """

        if not node_ids:
            return

        assert self._locked

        new_parent_path = self.get_path(parent_id)

        # Check if the new parent is below the node
        new_parent_set = set(new_parent_path)
        for node_id in node_ids:
            if node_id in new_parent_set:
                raise ValueError(
                    "Relocating {} to {} would create a circle".format(
                        node_id, parent_id
                    )
                )

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
        result = self._connection.execute(
            stmt, new_parent_id=parent_id, node_ids=tuple(node_ids)
        ).fetchall()

        # Invalidate subtree rooted at first common ancestor
        parent_paths = [new_parent_path] + [
            self.get_path(r["old_parent_id"]) for r in result
        ]
        paths_to_update = _paths_from_common_ancestor(parent_paths)
        nodes_to_invalidate = set(sum(paths_to_update, []))

        assert parent_id in nodes_to_invalidate

        self.invalidate_nodes(nodes_to_invalidate, unapprove)

    def merge_node_into(self, node_id, dest_node_id, unapprove=False):
        """
        Merge a node n into another node d.

        All objects of n will be assigned to d.
        All children of n will be assigned to d.
        n will be deleted.
        """

        # TODO: Invalidate all ancestors (until common root)

        # Change node for objects
        stmt = (
            # pylint: disable=no-value-for-parameter
            nodes_objects.update()
            .values(node_id=dest_node_id)
            .where(
                nodes_objects.c.node_id
                == node_id & nodes_objects.c.project_id
                == self.project_id
            )
        )
        self._connection.execute(stmt)

        # Change parent for children
        stmt = (
            # pylint: disable=no-value-for-parameter
            nodes.update()
            .values(parent_id=dest_node_id)
            .where(nodes.c.parent_id == node_id & nodes.c.project_id == self.project_id)
        )
        self._connection.execute(stmt)

        # TODO: Calculate path difference
        old_path = self.get_path(node_id)
        new_path = self.get_path(dest_node_id)
        sdiff = set(old_path).symmetric_difference(new_path)

        self.invalidate_nodes(sdiff, unapprove)

        # Delete node
        stmt = nodes.delete(nodes.c.node_id == node_id)
        self._connection.execute(stmt)

    def invalidate_nodes(self, nodes_to_invalidate, unapprove=False):
        """
        Invalidate the provided nodes.

        Args:
            nodes_to_invalidate: Collection of `node_id`s
            unapprove (bool): Unapprove the specified nodes.
        """

        assert self._locked

        values = {nodes.c.cache_valid: False}

        if unapprove:
            values[nodes.c.approved] = False

        stmt = (
            # pylint: disable=no-value-for-parameter
            nodes.update()
            .values(values)
            .where(nodes.c.node_id.in_(nodes_to_invalidate))
        )
        self._connection.execute(stmt)

    def get_node(self, node_id: int, require_valid=True):
        if require_valid:
            # TODO: Directly use values instead of reading again from DB
            self.consolidate_node(node_id)

        stmt = select([nodes]).where(
            nodes.c.node_id == node_id & nodes.c.project_id == self.project_id
        )

        result = self._connection.execute(stmt, node_id=node_id).fetchone()

        if result is None:
            raise ValueError("Node {} is unknown".format(node_id))

        return dict(result)

    def get_path(self, node_id: int):
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
                WHERE   n.node_id = :node_id AND n.project_id = :project_id
                UNION ALL
                SELECT  p.*, level + 1
                FROM    q
                JOIN    nodes AS p
                ON      p.node_id = q.parent_id
                WHERE   p.project_id = :project_id
            )
            SELECT  node_id
            FROM    q
            ORDER BY
            level DESC
        """
        )
        result = self._connection.execute(
            stmt, node_id=node_id, project_id=self.project_id
        )
        return [r for (r,) in result.fetchall()]

    def _update_node(self, node_id, op, v_sum=None):
        """
        Parameters:
            op: iadd or isub
        """

        assert self._locked

        # TODO: Gather current values
        v_sum_ = ...

        if v_sum is not None:
            op(v_sum_, v_sum)

    def delete(self):
        """Delete this project from the database."""
        assert self._locked
        raise NotImplementedError()

    def get_children(self, node_id, require_valid=True, order_by=None, include=None):
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

        assert self._locked

        if require_valid:
            self.consolidate_node(node_id, depth="children")

        stmt = select([nodes])

        stmt = stmt.where(nodes.c.parent_id == node_id)

        if include is not None:
            stmt = stmt.where(nodes.c.starred == (include == "starred"))

        if order_by is not None:
            stmt = stmt.order_by(text(order_by))

        result = self._connection.execute(stmt, node_id=node_id).fetchall()

        return [dict(r) for r in result]

    def get_objects(self, node_id, offset=None, limit=None, order_by=None):
        """
        Get objects directly below a node.
        """

        assert self._locked

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

        result = self._connection.execute(stmt, node_id=node_id).fetchall()

        return [dict(r) for r in result]

    def calculate_progress(self, node_id=None):
        """
        Calculate labeling progress.

        - Number of objects below approved nodes
        """

        if node_id is None:
            node_id = self.get_root_id()

        subtree = self.consolidate_node(
            node_id, depth="full", return_="raw", descend_approved=False
        )

        subtree["n_approved_objects"] = subtree["approved"] * subtree["_n_objects_deep"]
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

    def update_node(self, node_id, data):
        if "parent_id" in data:
            self.relocate_nodes([node_id], data.pop("parent_id"))

        assert "parent_id" not in data

        if "node_id" in data:
            raise ValueError("node_id is read-only")

        stmt = (
            nodes.update()  # pylint: disable=no-value-for-parameter
            .values(data)
            .where(nodes.c.node_id == node_id & nodes.c.project_id == self.project_id)
        )
        self._connection.execute(stmt)

    def reject_objects(self, node_id, object_ids):
        """
        Save objects as rejected for a certain node_id to prevent further recommendation.
        """

        assert self._locked

        if not object_ids:
            return

        new_node_path = self.get_path(node_id)

        # Update assignments
        # pylint: disable=no-value-for-parameter
        stmt = nodes_rejected_objects.insert()
        self._connection.execute(
            stmt,
            [
                {"node_id": node_id, "project_id": self.project_id, "object_id": oid}
                for oid in object_ids
            ],
        )

    def recommend_children(self, node_id, max_n=1000):
        node = self.get_node(node_id)

        # Get the path to the node
        path = self.get_path(node_id)

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

            # Get the path to the node (without the node itself)
            path = self.get_path(node_id)[:-1]

            objects_ = []

            rejected_object_ids = (
                select([nodes_rejected_objects.c.object_id])
                .where(
                    nodes_rejected_objects.c.node_id
                    == node_id & nodes_rejected_objects.c.project_id
                    == self.project_id
                )
                .alias("rejected_object_ids")
            )

            with timer.child("Query matching objects"):
                # Traverse the parse in reverse
                for parent_id in path[::-1]:
                    n_left = max_n - len(objects_)

                    # Break if we already have enough nodes
                    if n_left <= 0:
                        break

                    # Get objects below parent_id that are not rejected by node_id
                    stmt = (
                        select([objects])
                        .select_from(objects.join(nodes_objects))
                        .where(
                            (nodes_objects.c.node_id == parent_id)
                            & (~objects.c.object_id.in_(rejected_object_ids))
                            & nodes_objects.c.project_id
                            == self.project_id
                        )
                    )

                    result = self._connection.execute(stmt).fetchall()

                    objects_.extend(dict(r) for r in result)

            if not objects_:
                return []

            with timer.child("Convert to array"):
                vectors = [o["vector"] for o in objects_]
                objects_ = np.array(objects_, dtype=object)
                vectors = np.array(vectors)

            with timer.child("Calculate distances"):
                prots: Prototypes = node["_prototypes"]
                if prots is None:
                    raise ProjectError("Node has no prototypes!")

                distances = prots.transform(vectors)

            with timer.child("Sorting"):
                order = np.argsort(distances)[:max_n]

            with timer.child("Result assembly"):
                return objects_[order].tolist()

    def get_next_node(
        self, node_id, leaf=False, recurse_cb=None, filter=None, preferred_first=False
    ):
        """
        Get the id of the next unapproved node.

        This is either:
            a) the deepest node below, if this current node matches the recurse_cb, or
            b) the first node below a predecessor of the current node that does not match the recurse_cb.

        Parameters:
            node_id
            leaf: Only return leaves.
        """

        # First try if there are candidates below this node
        subtree = _rquery_subtree(node_id, recurse_cb)

        children = nodes.alias("children")
        n_children = (
            select([func.count()])
            .select_from(children)
            .where(
                (children.c.parent_id == subtree.c.node_id)
                & (children.c.project_id == self.project_id)
            )
            .as_scalar()
            .label("n_children")
        )

        stmt = select([subtree.c.node_id])

        if filter is not None:
            stmt = stmt.where(filter(subtree))

        if leaf:
            stmt = stmt.where(n_children == 0)

        if preferred_first:
            stmt = stmt.order_by(subtree.c.preferred.desc())

        stmt = stmt.order_by(subtree.c.level.desc()).limit(1)

        print(stmt)

        result = self._connection.execute(stmt).scalar()

        if result is not None:
            return result

        # Otherwise go to parent
        node = self._connection.execute(
            nodes.select(nodes.c.node_id == node_id)
        ).first()

        if node["parent_id"]:
            print("No unapproved children, trying parent: {}".format(node["parent_id"]))
            return self.get_next_node(node["parent_id"], leaf, recurse_cb, filter)

        return None

    # Context manager protocol

    def __enter__(self):
        self._transactions.append(self._connection.begin())

        try:
            # Advisory lock for project_id
            self._connection.execute(
                select([func.pg_advisory_xact_lock(self.project_id)])
            )
            self._locked += 1
        except:
            self._transactions.pop().rollback()
            raise

        return self

    def __exit__(self, type_, value, traceback):
        txn: Transaction = self._transactions.pop()

        if type_ is None and txn.is_active:
            txn.commit()
        else:
            txn.rollback()

        self._locked -= 1

    def remove(self):
        """Remove the project and all belonging entries from the database."""
        connection = database.get_connection()

        with connection.begin():
            # Drop partition for nodes_objects
            stmt = "DROP TABLE nodes_objects_{project_id} CASCADE".format(
                project_id=self.project_id
            )
            connection.execute(stmt)

            # Drop partition for nodes_rejected_objects
            stmt = "DROP TABLE nodes_rejected_objects_{project_id} CASCADE".format(
                project_id=self.project_id
            )
            connection.execute(stmt)

            # Delete entry
            stmt = projects.delete(projects.c.project_id == self.project_id)
            connection.execute(stmt)

        self.project_id = None

    def consolidate_node(self, node_id, depth=0, descend_approved=True, return_=None):
        """
        Ensure that the calculated values of this node are valid.

        If deep=True, ensures that also the calculated values of all successors are valid.

        Parameters:
            node_id: Root of the subtree that gets consolidated.
            depth: Ensure validity of cached values at least up to a certain depth.
            return_: None | "node" | "children". Return this node or its children.

        Returns:
            node dict or list of children, depending on return_ parameter.
        """

        if isinstance(depth, str):
            if depth == "children":
                depth = 1
            elif depth == "full":
                depth = -1
            else:
                raise NotImplementedError("Unknown depth string: {}".format(depth))

        connection = database.get_connection()

        assert self._locked

        # Wrap everything in a transaction
        with connection.begin():

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

            invalid_subtree: pd.DataFrame = pd.read_sql_query(
                stmt, connection, index_col="node_id"
            )

            if len(invalid_subtree) == 0:
                raise ProjectError("Unknown node: {}".format(node_id))

            if not invalid_subtree["cache_valid"].all():
                # 1. _n_objects, _n_children
                invalid_subtree["_n_objects"] = invalid_subtree["_n_objects_"]
                invalid_subtree["_n_children"] = invalid_subtree["_n_children_"]

                invalid_subtree["__updated"] = False

                # Iterate over DataFrame fixing the values along the way
                for node_id in tqdm.tqdm(invalid_subtree.index):
                    if invalid_subtree.at[node_id, "cache_valid"]:
                        # Don't recalculate valid nodes as invalid_subtree (rightly)
                        # doesn't include their children.
                        continue

                    child_selector = invalid_subtree["parent_id"] == node_id
                    children = invalid_subtree.loc[child_selector]
                    children_dict = MemberCollection(
                        children.reset_index().to_dict("records"), "zero"
                    )

                    # 2. _n_objects_deep
                    _n_objects = invalid_subtree.loc[node_id, "_n_objects"]
                    _n_objects_deep = _n_objects + children["_n_objects_deep"].sum()
                    invalid_subtree.at[node_id, "_n_objects_deep"] = _n_objects_deep

                    # Sample 1000 objects to speed up the calculation
                    objects_ = MemberCollection(
                        self.get_objects(node_id, order_by=objects.c.rand, limit=1000),
                        "raise",
                    )

                    # 3. _own_type_objects, _type_objects
                    # TODO: Replace _own_type_objects with "_atypical_objects"
                    invalid_subtree.at[
                        node_id, "_own_type_objects"
                    ] = self._calc_own_type_objects(children_dict, objects_)
                    # self._calc_own_type_objects(children_dict, objects_)

                    invalid_subtree.at[
                        node_id, "_type_objects"
                    ] = self._calc_type_objects(children_dict, objects_)

                    if (
                        children_dict
                        and len(invalid_subtree.at[node_id, "_type_objects"]) == 0
                    ):
                        print(
                            "\nNode {} has no type objects although it has children!".format(
                                node_id
                            )
                        )

                    # 4. _centroid
                    _centroid = []

                    if len(objects_) > 0:
                        # Object mean, weighted with number of objects
                        obj_mean = np.sum(objects_.vectors, axis=0)
                        obj_mean /= np.linalg.norm(obj_mean)
                        _centroid.append(_n_objects * obj_mean)
                    if len(children_dict) > 0:
                        children_mean = np.sum(
                            children_dict.cardinalities[:, np.newaxis]
                            * children_dict.vectors,
                            axis=0,
                        )
                        _centroid.append(children_mean)

                    if len(_centroid) > 0:
                        _centroid = np.sum(_centroid, axis=0)
                        _centroid /= np.linalg.norm(_centroid)
                    else:
                        _centroid = None

                    invalid_subtree.at[node_id, "_centroid"] = _centroid

                    if invalid_subtree.loc[node_id, "_centroid"] is None:
                        print("\nNode {} has no centroid!".format(node_id))

                    # 5. _prototypes
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
                            _prototypes = merge_prototypes(_prototypes, N_PROTOTYPES)
                        except:
                            for prots in _prototypes:
                                print(prots.prototypes_)
                            raise
                    else:
                        _prototypes = None
                        print("\nNode {} has no prototypes!".format(node_id))

                    invalid_subtree.at[node_id, "_prototypes"] = _prototypes

                    # Finally, flag as updated
                    invalid_subtree.at[node_id, "__updated"] = True

                    bar.numerator += 1
                    print(node_id, bar, end="    \r")
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
                return invalid_subtree[invalid_subtree["parent_id"] == node_id].to_dict(
                    "records"
                )

            if return_ == "raw":
                return invalid_subtree
