import contextlib
import itertools
import operator
import typing as T

import numpy as np
import pandas as pd
import pandas.api.types
import tqdm
from scipy.spatial.distance import cdist
from sqlalchemy import func, select, update
from sqlalchemy.sql import text
from sqlalchemy.sql.elements import ClauseElement, literal_column
from sqlalchemy.sql.expression import bindparam, literal
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
    datasets,
)


class ProjectError(Exception):
    pass


def commonprefix(paths):
    "Given a list of paths, return the longest common leading component, ignoring the last component."
    paths = [p[:-1] for p in paths]
    s1 = min(paths)
    s2 = max(paths)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


def is_scalar_na(v):
    if pandas.api.types.is_scalar(v):
        return pd.isna(v)
    return False


def is_scalar_zero(v):
    if pandas.api.types.is_scalar(v):
        return v == 0
    return False


def _paths_from_common_ancestor(paths):
    """
    Strip the common prefix (without the first common ancestor) from p1 and p2.
    """
    common_anestor_idx = len(commonprefix(paths)) - 1
    return [p[common_anestor_idx:] for p in paths]


def _rquery_subtree(project_id, node_id, recurse_cb=None):
    """
    Construct a selectable for the subtree rooted at node_id.

    `level` is 0 for the supplied `node_id` and increases for each level of
    hierarchy.

    Parameters:
        recurse_cb: A callback with two parameters (q, s). q is the recursive query, s is the successor.
            The callback must return a clause that can be used in where().
    """
    # Query
    q = (
        select([nodes, literal(0).label("level")])
        .where((nodes.c.node_id == node_id) & (nodes.c.project_id == project_id))
        .cte(recursive=True)
        .alias("q")
    )

    # Sub-node
    s = nodes.alias("s")

    # Recursive query
    rq = select([s, literal_column("level") + 1]).where(
        (s.c.parent_id == q.c.node_id) & (s.c.project_id == project_id)
    )

    # Conditional descent
    if callable(recurse_cb):
        condition = recurse_cb(q, s)
        rq = rq.where(condition)

    q = q.union_all(rq)

    return q


def _slice_center(length, n_items):
    start = max(0, (length - n_items) // 2)
    stop = start + n_items

    return slice(start, stop)


class Project:
    """
    Interact with projects.
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
    def get_all(dataset_id: int, visible_only=True):
        """Get a list of all projects in a dataset.
        """

        qroots = (
            select([nodes.c.project_id, nodes.c.node_id])
            .where(nodes.c.parent_id == None)
            .alias("roots")
        )
        qprojects = (
            select([projects, qroots.c.node_id])
            .select_from(
                projects.join(
                    qroots, qroots.c.project_id == projects.c.project_id, isouter=True
                )
            )
            .where(projects.c.dataset_id == dataset_id)
        )

        if visible_only:
            qprojects = qprojects.where(projects.c.visible == True)

        connection = database.get_connection()
        result = connection.execute(qprojects).fetchall()

        return [dict(r) for r in result]

    def __init__(self, project_id):
        self.project_id = project_id
        self._dataset_id = None

        self._connection = database.get_connection()

    @property
    def dataset_id(self):
        if self._dataset_id is not None:
            return self._dataset_id

        stmt = select([projects.c.dataset_id]).where(
            projects.c.project_id == self.project_id
        )
        dataset_id = self._connection.execute(stmt).scalar()

        if dataset_id is None:
            raise ProjectError("dataset_id is None")

        self._dataset_id = dataset_id

        return dataset_id

    def get_dataset(self):
        stmt = select([datasets]).where(
            projects.c.project_id == self.project_id
        ).limit(1)
        dataset = self._connection.execute(stmt).fetchone()

        if dataset is None:
            raise ProjectError("Dataset not found")

        return dict(dataset)


    @contextlib.contextmanager
    def lock(self):
        # Acquire exclusive transaction level advisory lock
        self._connection.execute(
            select([func.pg_advisory_xact_lock(self.project_id)])
        )

        yield _LockedProject(self)

        # The lock is cleared automatically when the transaction ends
        pass


class _LockedProject:
    # A locked project provides all the functionality that requires locking

    def __init__(self, project: Project):
        self.project = project

    def __getattr__(self, name):
        return getattr(self.project, name)

    def import_tree(self, tree):
        """
        Fill the project with the supplied tree.
        """

        print("Loading {}...".format(tree))

        # Bail if project already contains a tree
        stmt = (
            select([func.count()])
            .select_from(nodes)
            .where(nodes.c.project_id == self.project_id)
        )
        if self._connection.execute(stmt).scalar() != 0:
            raise ProjectError(
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

        dataset_id = self.dataset_id

        # Insert object_ids
        if object_ids is not None:
            object_ids = iter(object_ids)
            while True:
                chunk = itertools.islice(object_ids, 1000)

                data = [
                    dict(
                        project_id=self.project_id,
                        node_id=node_id,
                        object_id=object_id,
                        dataset_id=dataset_id,
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

        if root_id is None:
            root_id = self.get_root_id()

        # Get complete subtree with up to date cached values

        print("Consolidating cached values...")
        subtree = self.consolidate_node(
            root_id, depth="full", return_="raw", exact_vector="approx"
        )

        keep_columns = [
            "parent_id",
            "name",
            "preferred",
            "filled",
            "approved",
            "n_children_",
            "n_objects_own_",
            "n_objects_",
            "n_approved_objects_",
            "n_approved_nodes_",
            "n_filled_objects_",
            "n_filled_nodes_",
            "n_preferred_objects_",
            "n_preferred_nodes_",
        ]
        tree_nodes = subtree[keep_columns].reset_index()

        print("Getting objects...")
        # Get subtree below root
        subtree = _rquery_subtree(self.project_id, root_id)

        # Get object IDs for all nodes
        node_objects = (
            select([nodes_objects.c.node_id, nodes_objects.c.object_id])
            .select_from(nodes_objects)
            .where(nodes_objects.c.project_id == self.project_id)
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

        stmt = select([nodes.c.node_id]).where(
            (nodes.c.parent_id == None) & (nodes.c.project_id == self.project_id)
        )
        root_id = self._connection.execute(stmt).scalar()

        if root_id is None:
            raise ValueError("Project has no root.")

        return root_id

    def to_dict(self) -> T.Dict:
        """Get the properties of the project as a dict.
        """

        qroots = (
            select([nodes.c.project_id, nodes.c.node_id])
            .where(nodes.c.parent_id == None)
            .alias("roots")
        )
        qprojects = (
            select([projects, qroots.c.node_id])
            .select_from(
                projects.join(
                    qroots, qroots.c.project_id == projects.c.project_id, isouter=True
                )
            )
            .where(projects.c.project_id == self.project_id)
            .limit(1)
        )

        print(qprojects)

        result = self._connection.execute(qprojects).fetchone()

        assert result is not None

        return dict(result)

    def relocate_objects(self, object_ids_or_condition, new_node_id):
        # https://stackoverflow.com/questions/7923237/return-pre-update-column-values-using-sql-only-postgresql-version
        # UPDATE node_objects x
        # SET    node_id = :new_node_id
        # FROM   node_objects y JOIN objects z ON x.object_id=z.object.id
        # WHERE  x.object_id = y.object_id
        # AND    x.object_id IN :object_ids
        # AND    x.project_id = :project_id AND y.project_id=:project_id
        # RETURNING y.node_id AS old_node_id, z.vector ORDER BY old_node_id;

        if isinstance(object_ids_or_condition, ClauseElement):
            condition = object_ids_or_condition
        else:
            object_ids = list(object_ids_or_condition)
            if not object_ids:
                return
            condition = nodes_objects.c.object_id.in_(object_ids)

        old_node_objects = nodes_objects.alias("old")
        updater = (
            update(nodes_objects)
            .values(node_id=new_node_id)
            .where(
                condition
                & (nodes_objects.c.project_id == self.project_id)
                & (old_node_objects.c.project_id == self.project_id)
                & (old_node_objects.c.object_id == nodes_objects.c.object_id)
                & (objects.c.dataset_id == self.dataset_id)
                & (objects.c.object_id == nodes_objects.c.object_id)
            )
            .returning(old_node_objects.c.node_id, objects.c.vector)
        ).cte("updater")

        # Order by node_id so that itertools.groupby works as expected
        stmt = select([updater]).order_by(updater.c.node_id)

        rows = self._connection.execute(stmt).fetchall()

        if not rows:
            return 0

        new_path = self.get_path(new_node_id)

        affected_precursers = set(new_path)

        # Calculate update values for old nodes
        old_node_ids = []
        old_vectors = []
        old_counts = []
        old_paths = []
        for old_node_id, old_rows in itertools.groupby(rows, operator.itemgetter(0)):
            old_node_ids.append(old_node_id)
            vectors = [r[1] for r in old_rows]
            vectors = np.array(vectors)
            old_vectors.append(np.sum(vectors, axis=0))
            old_counts.append(len(vectors))
            old_path = self.get_path(old_node_id)
            old_paths.append(old_path)
            affected_precursers.update(old_path)

        # Calculate common prefix of all paths
        prefix = commonprefix(old_paths + [new_path])
        len_prefix = len(prefix)

        new_vector = np.sum(old_vectors, axis=0)
        assert new_vector is not None

        # Update new node
        self.update_cached_values(
            [new_node_id],
            "+",
            n_objects_own_=len(rows),
            vector_sum_own_=new_vector,
            type_objects_own_=None,
            vector_mean_own_=None,
        )

        # Update new tree (this includes the new node)
        self.update_cached_values(
            new_path[len_prefix:],
            "+",
            n_objects_=len(rows),
            vector_sum_=new_vector,
            type_objects_=None,
            vector_mean_=None,
        )

        # Update old
        for old_node_id, old_vector, old_count, old_path in zip(
            old_node_ids, old_vectors, old_counts, old_paths
        ):
            assert old_vector is not None

            # Update old node
            self.update_cached_values(
                [old_node_id],
                "-",
                n_objects_own_=old_count,
                vector_sum_own_=old_vector,
                type_objects_own_=None,
                vector_mean_own_=None,
            )

            # Update old tree (this includes old node)
            update_old_node_ids = old_path[len_prefix:]
            self.update_cached_values(
                update_old_node_ids,
                "-",
                n_objects_=old_count,
                vector_sum_=old_vector,
                type_objects_=None,
                vector_mean_=None,
            )

        # Update flag summary in affected precursors
        self.update_cached_values(
            affected_precursers,
            "=",
            n_approved_objects_=None,
            n_filled_objects_=None,
            n_preferred_objects_=None,
            vector_mean_=None,
        )

        return len(rows)

    def update_cached_values(self, node_ids, op, **values):

        if op == "=":
            # Cache is invalidated if None in values
            if None in values.values():
                values["cache_valid"] = False
            stmt = (
                # pylint: disable=no-value-for-parameter
                nodes.update()
                .values(values)
                .where(
                    (nodes.c.project_id == self.project_id)
                    & (nodes.c.node_id.in_(node_ids))
                )
            )
            self._connection.execute(stmt)
            return

        try:
            op = {"+": operator.add, "-": operator.sub}[op]
        except KeyError:
            raise ValueError("Unexpected op: {!r}".format(op)) from None

        # Get current values
        columns = [nodes.c[c] for c in values.keys()]
        stmt = select([nodes.c.node_id.label("_node_id")] + columns).where(
            (nodes.c.project_id == self.project_id) & (nodes.c.node_id.in_(node_ids))
        )
        rows = self._connection.execute(stmt).fetchall()

        if not rows:
            return

        def _calc_new_value(row, k, v):
            if is_scalar_na(v):
                # None means reset to None, regardless of old value
                return None

            if is_scalar_na(row[k]):
                # TODO
                # # NA means vec(0) for vectors
                # if k in ("vector_sum_", "vector_sum_own_"):
                #     return op(0, v)
                # raise ValueError("row[{}] is NA.".format(k))
                # We can't update a NA
                return None

            return op(row[k], v)

        def _cleanup_vectors(row):
            """Reset "vector_sum_", "vector_sum_own_" to 0 if n_objects[_own]_ is 0."""
            if "n_objects_own_" in row and row["n_objects_own_"] == 0:
                row["vector_sum_own_"] = 0
            if "n_objects_" in row and row["n_objects_"] == 0:
                row["vector_sum_"] = 0
            return row

        # Update values
        rows = [
            _cleanup_vectors(
                dict(
                    ((k, _calc_new_value(row, k, v)) for k, v in values.items()),
                    _node_id=row["_node_id"],
                    # Invalidate unconditionally as _cleanup_vectors may set some values to None
                    cache_valid=False,
                )
            )
            for row in rows
        ]

        # Write updated values
        columns.append("cache_valid")
        stmt = (
            # pylint: disable=no-value-for-parameter
            nodes.update()
            .where(nodes.c.node_id == bindparam("_node_id"))
            .values({c: bindparam(c) for c in columns})
        )
        # print("update_cached_values:rows:", rows)
        self._connection.execute(stmt, rows)

    def relocate_nodes(self, node_ids_or_condition, new_parent_id):
        """
        Relocate nodes to another parent.

        Returns:
            int: Number of nodes relocated.
        """

        if isinstance(node_ids_or_condition, ClauseElement):
            condition = node_ids_or_condition
        else:
            if not node_ids_or_condition:
                return
            condition = nodes.c.node_id.in_(node_ids_or_condition)

        new_parent_path = self.get_path(new_parent_id)

        old_nodes = nodes.alias("old")
        updater = (
            update(nodes)
            .values(parent_id=new_parent_id)
            .where(
                condition
                & (nodes.c.project_id == self.project_id)
                & (~nodes.c.node_id.in_(new_parent_path))  # Prevent cycles
                & (old_nodes.c.project_id == self.project_id)
                & (old_nodes.c.node_id == nodes.c.node_id)
            )
            .returning(
                old_nodes.c.parent_id,
                old_nodes.c.node_id,
                old_nodes.c.n_objects_,
                old_nodes.c.n_nodes_,
                old_nodes.c.vector_sum_,
            )
        ).cte("updater")

        # Order by parent_id so that itertools.groupby works as expected
        stmt = select([updater]).order_by(updater.c.parent_id)

        rows = self._connection.execute(stmt).fetchall()

        if not rows:
            return 0

        print("relocate_nodes: {} nodes will be relocated.".format(len(rows)))

        affected_precursers = set(new_parent_path)

        # Calculate update values for old parents
        old_parent_ids = []
        old_parent_paths = []
        old_n_objectss = []
        old_n_nodess = []
        old_vector_sums = []
        old_n_childrens = []

        for old_parent_id, old_rows in itertools.groupby(
            rows, operator.itemgetter(updater.c.parent_id)
        ):
            old_rows = list(old_rows)

            old_n_childrens.append(len(old_rows))

            old_parent_ids.append(old_parent_id)
            old_parent_path = self.get_path(old_parent_id)
            old_parent_paths.append(old_parent_path)

            # Add nodes from old_parent_path to affected_precursers
            affected_precursers.update(old_parent_path)

            old_n_objects = sum(r[updater.c.n_objects_] for r in old_rows)
            old_n_objectss.append(old_n_objects)

            old_n_nodes = sum(r[updater.c.n_nodes_] for r in old_rows)
            old_n_nodess.append(old_n_nodes)

            vectors = [r[updater.c.vector_sum_] for r in old_rows]
            vectors = np.array(vectors)
            old_vector_sums.append(np.sum(vectors, axis=0))

            # Also add node_ids to affected_precursers
            affected_precursers.update(r[updater.c.node_id] for r in old_rows)

        # Calculate common prefix of all paths
        prefix = commonprefix(old_parent_paths + [new_parent_path])
        len_prefix = len(prefix)

        new_vector_sum = np.sum(old_vector_sums, axis=0)
        new_n_objects = sum(old_n_objectss)
        new_n_nodes = sum(old_n_nodess)
        assert new_vector_sum is not None

        # Update new parent
        self.update_cached_values([new_parent_id], "+", n_children_=len(rows))

        # Update new tree (this includes the new parent)
        self.update_cached_values(
            new_parent_path[len_prefix:],
            "+",
            n_nodes_=new_n_nodes,
            n_objects_=new_n_objects,
            vector_sum_=new_vector_sum,
            type_objects_=None,
        )

        # Update old
        for (
            old_parent_id,
            old_n_nodes,
            old_n_objects,
            old_parent_path,
            old_vector_sum,
            old_n_children,
        ) in zip(
            old_parent_ids,
            old_n_nodess,
            old_n_objectss,
            old_parent_paths,
            old_vector_sums,
            old_n_childrens,
        ):
            assert old_vector_sum is not None

            # Update old parent
            self.update_cached_values([old_parent_id], "-", n_children_=old_n_children)

            # Update old tree (this includes old parent)
            self.update_cached_values(
                old_parent_path[len_prefix:],
                "-",
                n_nodes_=None,  # old_n_nodes, # Can not be reliably calculated.
                n_objects_=old_n_objects,
                vector_sum_=old_vector_sum,
                type_objects_=None,
            )

        # Update affected precursors (including self)
        print(
            "relocate_nodes: Resetting {} affected precursers...".format(
                len(affected_precursers)
            )
        )
        self.update_cached_values(
            affected_precursers,
            "=",
            n_approved_objects_=None,
            n_filled_objects_=None,
            n_preferred_objects_=None,
            n_approved_nodes_=None,
            n_filled_nodes_=None,
            n_preferred_nodes_=None,
            vector_mean_=None,
        )

        return len(rows)

    def merge_node_into(self, node_id, dest_node_id):
        """
        Merge a node n into another node d.

        All objects of n will be assigned to d.
        All children of n will be assigned to d.
        n will be deleted.

        Returns:
            tuple: (n_relocated_objects, n_relocated_children)
        """

        node_path = self.get_path(node_id)

        # Relocate objects
        n_relocated_objects = self.relocate_objects(
            nodes_objects.c.node_id == node_id, dest_node_id
        )

        # Relocate children
        n_relocated_children = self.relocate_nodes(
            nodes.c.parent_id == node_id, dest_node_id
        )

        # Delete node
        stmt = nodes.delete(
            (nodes.c.node_id == node_id) & (nodes.c.project_id == self.project_id)
        )
        self._connection.execute(stmt)

        # Update n_children_ of parent
        self.update_cached_values([node_path[-2]], "-", n_children_=1)

        # Update cached values of precursors
        self.update_cached_values(
            node_path[:-1],
            "-",
            n_nodes_=1,
            n_approved_objects_=None,
            n_filled_objects_=None,
            n_preferred_objects_=None,
            n_approved_nodes_=None,
            n_filled_nodes_=None,
            n_preferred_nodes_=None,
        )

        return (n_relocated_objects, n_relocated_children)

    def invalidate_nodes(self, nodes_to_invalidate, unapprove=False):
        """
        Invalidate the provided nodes.

        Args:
            nodes_to_invalidate: Collection of `node_id`s
            unapprove (bool): Unapprove the specified nodes.
        """

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
            (nodes.c.node_id == node_id) & (nodes.c.project_id == self.project_id)
        )

        result = self._connection.execute(stmt, node_id=node_id).fetchone()

        if result is None:
            raise ValueError("Node {} is unknown".format(node_id))

        return dict(result)

    def get_all_nodes(self, require_valid=True):
        root_id = self.get_root_id()

        if require_valid:
            subtree = self.consolidate_node(root_id, depth=-1, return_="raw")
        else:
            raise NotImplementedError()

        subtree.replace({np.nan: None}, inplace=True)

        return subtree.reset_index().to_dict(orient="records")

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
        """Get objects directly below a node."""

        stmt = (
            select([objects])
            .select_from(objects.join(nodes_objects))
            .where(
                (nodes_objects.c.node_id == node_id)
                & (nodes_objects.c.project_id == self.project_id)
                & (nodes_objects.c.dataset_id == self.dataset_id)
            )
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

        subtree = self.consolidate_node(node_id, depth="full", return_="raw")

        fields = [
            "n_nodes_",
            "n_objects_",
            "n_approved_objects_",
            "n_approved_nodes_",
            "n_filled_objects_",
            "n_filled_nodes_",
            "n_preferred_objects_",
            "n_preferred_nodes_",
        ]

        # Leaves
        leaves_mask = subtree["n_children_"] == 0
        leaves = subtree.loc[leaves_mask, fields]
        print(leaves)
        leaves_result = leaves.sum(axis=0).to_dict()
        leaves_result = {
            "leaves_{}".format(k.rstrip("_")): v for k, v in leaves_result.items()
        }

        deep_result = subtree.loc[node_id, fields].to_dict()
        deep_result = {k.rstrip("_"): v for k, v in deep_result.items()}

        return dict(**leaves_result, **deep_result)

    def reject_objects(self, node_id, object_ids):
        """
        Save objects as rejected for a certain node_id to prevent further recommendation.
        """

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
                    (nodes_rejected_objects.c.node_id == node_id)
                    & (nodes_rejected_objects.c.project_id == self.project_id)
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
                            & (nodes_objects.c.project_id == self.project_id)
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
        self, node_id=None, leaf=False, recurse_cb=None, filter=None, preferred_first=False
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

        if node_id is None:
            node_id = self.get_root_id()

        # First try if there are candidates below this node
        subtree = _rquery_subtree(self.project_id, node_id, recurse_cb)

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

    def get_subtree(self, node_id, recurse_cb=None, **replace_columns) -> pd.DataFrame:
        """
        Get the subtree below node_id as a DataFrame.

        Args:
            node_id: Root of the subtree
            recurse_cb: Callback for recursion. See _rquery_subtree.
            **replace_columns: Replace stored values with other values (e.g. calculated).

        Returns:
            pd.DataFrame
        """

        invalid_subtree = _rquery_subtree(self.project_id, node_id, recurse_cb)

        columns = {c.key: c for c in invalid_subtree.c}
        columns.update(
            {
                k: v(invalid_subtree) if callable(v) else v
                for k, v in replace_columns.items()
            }
        )

        stmt = (
            select(columns.values())
            # Deepest nodes first
            .order_by(invalid_subtree.c.level.desc())
        )

        invalid_subtree: pd.DataFrame = pd.read_sql_query(
            stmt, self._connection, index_col="node_id"
        )

        if not len(invalid_subtree):  # pylint: disable=len-as-condition
            raise ProjectError("Unknown node: {}".format(node_id))

        return invalid_subtree

    def consolidate_node(
        self,
        node_id,
        depth=0,
        descend_approved=True,
        exact_vector="raise",
        return_=None,
    ):
        """
        Ensure that the calculated values of this node are valid.

        If deep=True, ensures that also the calculated values of all successors are valid.

        Parameters:
            node_id: Root of the subtree that gets consolidated.
            depth: Ensure validity of cached values at least up to a certain depth.
                -1 means full subtree.
            exact_vector (str, "exact"|"raise"|"approx"):
                exact: Calculate vector as the sum of all objects.
                raise: Do not recalculate vector and raise an exception instead.
                approx: Calculate approximation from a subsample of objects.
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

        if exact_vector not in ("exact", "raise", "approx"):
            raise ValueError("exact_vector has to be one of exact, raise, approx")

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

        # Readily query real n_objects
        def n_objects_own_(invalid_subtree):
            return (
                select([func.count()])
                .select_from(nodes_objects)
                .where(nodes_objects.c.project_id == self.project_id)
                .where(nodes_objects.c.node_id == invalid_subtree.c.node_id)
                .as_scalar()
                .label("n_objects_own_")
            )

        # Readily query real n_children
        def n_children_(invalid_subtree):
            children = nodes.alias("children")
            return (
                select([func.count()])
                .select_from(children)
                .where(children.c.project_id == self.project_id)
                .where(children.c.parent_id == invalid_subtree.c.node_id)
                .as_scalar()
                .label("n_children_")
            )

        invalid_subtree = self.get_subtree(
            node_id, recurse_cb, n_children_=n_children_, n_objects_own_=n_objects_own_
        )

        print("type n_objects_:", invalid_subtree["n_objects_"].dtype)
        print("unique n_objects_:", invalid_subtree["n_objects_"].unique())

        if not invalid_subtree["cache_valid"].all():
            invalid_subtree["updated__"] = False

            # Iterate over DataFrame fixing the values along the way (deepest nodes)
            with tqdm.tqdm(invalid_subtree.index) as t:
                for node_id in t:
                    # t.set_description("Consolidating {}".format(node_id))

                    n: dict = invalid_subtree.loc[node_id].to_dict()
                    updated_names = ["n_children_", "n_objects_own_"]

                    if n["cache_valid"]:
                        # Don't recalculate valid nodes as invalid_subtree (rightly)
                        # doesn't include their children.
                        continue

                    child_selector = invalid_subtree["parent_id"] == node_id
                    children = invalid_subtree.loc[child_selector]

                    children_collection = MemberCollection(
                        children.reset_index().to_dict("records"), "zero"
                    )

                    ## n_children_
                    # Is already present as calculated by the database

                    ## n_nodes_
                    if pd.isnull(n["n_nodes_"]):
                        assert not pd.isna(children["n_nodes_"]).any()
                        n["n_nodes_"] = 1 + children["n_nodes_"].sum()
                        updated_names.append("n_nodes_")

                    ## n_objects_own_
                    # Is already present as calculated by the database

                    ## n_objects_
                    if pd.isnull(n["n_objects_"]):
                        n["n_objects_"] = (
                            n["n_objects_own_"] + children["n_objects_"].sum()
                        )
                        updated_names.append("n_objects_")

                    N_OBJECTS_SAMPLE = 1000

                    # Sample 1000 objects to speed up the calculation
                    if (
                        n["vector_sum_own_"] is None or n["type_objects_own_"] is None
                    ) and n["n_objects_"]:
                        limit = None if exact_vector == "exact" else N_OBJECTS_SAMPLE
                        objects_sample = self.get_objects(
                            node_id, order_by=objects.c.rand, limit=limit
                        )
                    else:
                        objects_sample = []

                    objects_sample = MemberCollection(objects_sample, "raise")
                    n_objects_sample = len(objects_sample)  # actual sample size
                    objects_sample_vectors = objects_sample.vectors

                    ## vector_sum_own_
                    if n["vector_sum_own_"] is None:
                        if exact_vector == "raise":
                            raise ProjectError("vector_sum_own_ is not set")

                        if n_objects_sample > 0:
                            n["vector_sum_own_"] = np.sum(
                                objects_sample_vectors, axis=0
                            )

                            if n_objects_sample < n["n_objects_own_"]:
                                # If the sample size is smaller than the actual number of objects, we need to account for that
                                n["vector_sum_own_"] *= (
                                    n["n_objects_own_"] / n_objects_sample
                                )
                        else:
                            n["vector_sum_own_"] = 0

                        updated_names.append("vector_sum_own_")

                    ## vector_sum_
                    if n["vector_sum_"] is None:
                        if n["n_objects_"] > 0:
                            n["vector_sum_"] = n["vector_sum_own_"] + np.sum(
                                children["vector_sum_"], axis=0
                            )
                        else:
                            n["vector_sum_"] = 0

                        updated_names.append("vector_sum_")

                    ## vector_mean_own_
                    if n["vector_mean_own_"] is None and n["n_objects_own_"] > 0:
                        n["vector_mean_own_"] = (
                            n["vector_sum_own_"] / n["n_objects_own_"]
                        )
                        updated_names.append("vector_mean_own_")

                    ## vector_mean_
                    if n["vector_mean_"] is None and n["n_objects_"] > 00:
                        n["vector_mean_"] = n["vector_sum_"] / n["n_objects_"]
                        updated_names.append("vector_mean_")

                    ## type_objects_own_
                    N_TYPE_OBJECTS = 9
                    if n["type_objects_own_"] is None and n_objects_sample:
                        vector_sum_own_mean = n["vector_sum_own_"] / n_objects_sample
                        # Order objects_sample_vectors by distance to vector_sum_own_mean
                        sqdistances = cdist(
                            vector_sum_own_mean[np.newaxis, :],
                            objects_sample_vectors,
                            "sqeuclidean",
                        )[0]
                        ordered_idx = np.argsort(sqdistances)
                        # Find central N_TYPE_OBJECTS
                        central_idx = ordered_idx[
                            _slice_center(len(ordered_idx), N_TYPE_OBJECTS)
                        ]

                        n["type_objects_own_"] = [
                            objects_sample[i]["object_id"] for i in central_idx
                        ]
                        updated_names.append("type_objects_own_")

                    ## type_objects_
                    if n["type_objects_"] is None:
                        type_pool = []
                        if n["type_objects_own_"] is not None:
                            type_pool.extend(n["type_objects_own_"])

                        for cto in children["type_objects_"]:
                            if cto is not None:
                                type_pool.extend(cto)

                        if type_pool:
                            n["type_objects_"] = np.random.choice(
                                type_pool, min(len(type_pool), 9), replace=False
                            )
                            updated_names.append("type_objects_")

                    ## n_(approved|filled|preferred)_objects_
                    ## n_(approved|filled|preferred)_nodes_
                    for flag in ("approved", "filled", "preferred"):
                        n_X_objects_name = "n_{}_objects_".format(flag)
                        n_X_nodes_name = "n_{}_nodes_".format(flag)

                        if pd.isnull(n[n_X_objects_name]):
                            n[n_X_objects_name] = (
                                n[flag] * n["n_objects_own_"]
                                + children[n_X_objects_name].sum()
                            )
                            updated_names.append(n_X_objects_name)

                        if pd.isnull(n[n_X_nodes_name]):
                            n[n_X_nodes_name] = (
                                int(n[flag]) + children[n_X_nodes_name].sum()
                            )
                            updated_names.append(n_X_nodes_name)

                    ## Finally, flag as updated
                    n["updated__"] = True
                    updated_names.append("updated__")

                    # Write values back
                    updated_values = tuple(n[name] for name in updated_names)
                    invalid_subtree.loc[node_id, updated_names] = updated_values

            # Convert n_objects_own_ to int (might be object when containing NULL values in the database)
            invalid_subtree["n_objects_own_"] = invalid_subtree[
                "n_objects_own_"
            ].astype(int)

            invalid_subtree["n_nodes_"] = invalid_subtree["n_nodes_"].astype(int)

            # Flag all rows as valid
            invalid_subtree["cache_valid"] = True

            # Mask for updated rows
            updated_selection = invalid_subtree["updated__"] == True
            n_updated = updated_selection.sum()

            # Write back to database (if necessary)
            if n_updated > 0:
                # Write results back to database
                update_fields = [
                    "cache_valid",
                    "n_children_",
                    "n_nodes_",
                    "n_objects_own_",
                    "n_objects_",
                    "vector_sum_own_",
                    "vector_sum_",
                    "vector_mean_own_",
                    "vector_mean_",
                    "type_objects_own_",
                    "type_objects_",
                    "n_approved_objects_",
                    "n_approved_nodes_",
                    "n_filled_objects_",
                    "n_filled_nodes_",
                    "n_preferred_objects_",
                    "n_preferred_nodes_",
                ]

                stmt = (
                    # pylint: disable=no-value-for-parameter
                    nodes.update()
                    .where(nodes.c.node_id == bindparam("_node_id"))
                    .values({k: bindparam(k) for k in update_fields})
                )

                # Build the result list of dicts with _node_id and only update_fields
                result = invalid_subtree.loc[updated_selection, update_fields]
                result.index.rename("_node_id", inplace=True)
                result.reset_index(inplace=True)

                self._connection.execute(stmt, result.to_dict("records"))

                print("Updated {:d} nodes.".format(n_updated))

            # Drop updated__ column again to be compatible with get_subtree
            invalid_subtree.drop(columns=["updated__"], inplace=True)

        if return_ == "node":
            return invalid_subtree.loc[node_id].to_dict()

        if return_ == "children":
            return invalid_subtree[invalid_subtree["parent_id"] == node_id].to_dict(
                "records"
            )

        if return_ == "raw":
            return invalid_subtree

    def reset_cached_values(self):

        # Cached values are prefixed with an underscore
        values = {
            c: None for c in nodes.c.keys() if c.endswith("_")
        }  # pylint: disable=no-value-for-parameter
        values["cache_valid"] = False
        stmt = (
            # pylint: disable=no-value-for-parameter
            nodes.update()
            .values(values)
            .where(nodes.c.project_id == self.project_id)
        )
        self._connection.execute(stmt)

    def get_n_objects(self):

        stmt = (
            select([func.count()])
            .select_from(nodes_objects)
            .where(
                (nodes_objects.c.project_id == self.project_id)
                & (nodes_objects.c.dataset_id == self.dataset_id)
            )
        )

        n_objects = self._connection.execute(stmt).scalar()

        return n_objects

    def get_n_nodes(self):

        stmt = (
            select([func.count()])
            .select_from(nodes)
            .where(nodes.c.project_id == self.project_id)
        )

        n_nodes = self._connection.execute(stmt).scalar()

        return n_nodes

    def update_node(self, node_id, data):

        allowed_fields = {
            "parent_id",
            "name",
            "starred",
            "approved",
            "filled",
            "preferred",
        }
        if any(k not in allowed_fields for k in data):
            raise ProjectError(
                "Only {} can be updated".format(", ".join(allowed_fields))
            )

        if "parent_id" in data:
            self.relocate_nodes([node_id], data.pop("parent_id"))

        assert "parent_id" not in data

        path = self.get_path(node_id)

        old_nodes = nodes.alias("old_nodes")

        stmt = (
            nodes.update()  # pylint: disable=no-value-for-parameter
            .values(data)
            .where(
                (nodes.c.node_id == node_id)
                & (nodes.c.project_id == self.project_id)
                & (old_nodes.c.project_id == self.project_id)
                & (old_nodes.c.node_id == nodes.c.node_id)
            )
            .returning(old_nodes)
        )

        result = self._connection.execute(stmt)

        old_data = result.fetchone()

        # Make sure we only got one row
        assert result.fetchone() is None

        n_objects_ = old_data[old_nodes.c.n_objects_]

        for flag_name in set(("approved", "filled", "preferred")).intersection(data.keys()):
            direction = data[flag_name] - old_data[flag_name]
            if direction == 0:
                # No change
                continue

            op = {-1: "-", 1: "+"}[direction]
            values = {
                "n_{}_nodes_".format(flag_name): 1,
                "n_{}_objects_".format(flag_name): n_objects_,
            }
            self.update_cached_values(path, op, **values)

    def consolidate(self):

        root_id = self.get_root_id()

        self.consolidate_node(root_id, depth=-1, exact_vector="exact")
