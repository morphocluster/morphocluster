'''
Created on 13.03.2018

@author: mschroeder
'''
from morphocluster.models import nodes, projects, nodes_objects, objects
from sqlalchemy.sql import text
from sqlalchemy.sql.expression import select, outerjoin
import os
import pandas as pd
from etaprogress.progress import ProgressBar
from sqlalchemy.sql.functions import coalesce, func
from sqlalchemy.exc import SQLAlchemyError
import itertools
from numbers import Integral
import numpy as np
import warnings
from morphocluster.classifier import Classifier
from morphocluster.extensions import database
import csv
from genericpath import commonprefix


class TreeError(Exception):
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


class Tree(object):
    def __init__(self, connection):
        self.connection = connection
        
        
    def load_project(self, name, path):
        tree_fn = os.path.join(path, "tree.csv")
        objids_fn = os.path.join(path, "objids.csv")
        
        raw_tree = pd.read_csv(tree_fn, index_col=False)
        
        objids = pd.read_csv(objids_fn, index_col=False, names=["objid"], header=None, squeeze = True)
        
        raw_tree_nodes = raw_tree[raw_tree["child_size"] > 1]
        raw_tree_objects = raw_tree[raw_tree["child_size"] == 1]
        
        del raw_tree
        
        raw_tree_nodes.sort_values("parent", inplace = True)
        
        root_orig_id = int(raw_tree_nodes["parent"].iloc[0])
        
        with self.connection.begin():
            project_id = self.create_project(name)
            root_node_id = self.create_node(project_id, root_orig_id)
            
            print("Created root (orig/real):", root_orig_id, root_node_id)
        
            bar = ProgressBar(len(raw_tree_objects), max_width=40)
            
            for row in raw_tree_nodes.itertuples():
                # row.child is the current node
                # row.parent is its parent
                
                # Get object ids for the current node
                object_idxs = raw_tree_objects[raw_tree_objects["parent"] == row.child]["child"]
                object_ids = objids[object_idxs]

                self.create_node(project_id,
                                 orig_node_id = row.child,
                                 orig_parent = row.parent,
                                 object_ids = object_ids)
                    
                bar.numerator += len(object_ids)
                
                print(bar, end="\r")
            print()
            
        return project_id
    
    def get_objects_recursive(self, node_id):
        # Recursively select all descendants
        rquery = select([nodes]).where(nodes.c.node_id == node_id).cte(recursive=True)
        
        parents = rquery.alias("n")
        descendants = nodes.alias("nd")
        
        rquery = rquery.union_all(
            select([descendants]).where(descendants.c.parent_id == parents.c.node_id))
        
        # For each node in rquery, get associated objects
        obj_query = select([objects]).distinct().\
            select_from(rquery.join(nodes_objects).join(objects))
            
        result = self.connection.execute(obj_query)
        
        return [dict(r) for r in result]
    
    
    def export_classifications(self, root_id, classification_fn):
        """
        Export `object_id`s with cluster labels.
        """
        
        with self.connection.begin(), open(classification_fn, "w") as f:
            writer = csv.writer(f, delimiter=",")
            starred_nodes = self.get_minlevel_starred(root_id, cache_valid = False)
            
            bar = ProgressBar(len(starred_nodes), max_width=40)
            
            for node in starred_nodes:
                objs = self.get_objects_recursive(node["node_id"])
                
                for o in objs:
                    writer.writerow((o["object_id"], node["name"]))
                    
                bar.numerator += 1
                print(bar, end="\r")


    def get_root_id(self, project_id):
        """
        """
        stmt = select([nodes.c.node_id], (nodes.c.parent_id == None) & (nodes.c.project_id == project_id))
        root_id = self.connection.execute(stmt).scalar()
        
        if root_id is None:
            raise TreeError("No root")
        
        return root_id
    
    
    def get_projects(self):
        """
        Get projects with name
        """
        
        stmt = text("""
        SELECT p.*, n.node_id
        FROM nodes AS n
        JOIN projects AS p
        ON p.project_id = n.project_id
        WHERE n.parent_id IS NULL
        """)
        
        result = self.connection.execute(stmt).fetchall()
        
        return [dict(r) for r in result]
    
    
    def get_project(self, project_id):
        """
        Get a project by its ID.
        """
        
        stmt = text("""
        SELECT p.*, n.node_id
        FROM nodes AS n
        JOIN projects AS p
        ON p.project_id = n.project_id
        WHERE n.parent_id IS NULL AND p.project_id = :project_id
        """)
        
        result = self.connection.execute(stmt, project_id = project_id).fetchone()
        
        return dict(result)
    
    
    def get_path_ids(self, node_id):
        """
        Get the path of the node.
        
        Returns:
            List of `node_id`s.
        """
        stmt = text("""
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
        """)
        rows = self.connection.execute(stmt, node_id=node_id).fetchall()
        return [ r for (r,) in rows ]
    
    
    def create_project(self, name):
        """
        Create a project with a name and return its id.
        """
        stmt = projects.insert({"name": name})
        result = self.connection.execute(stmt)
        project_id = result.inserted_primary_key[0]
        
        return project_id
    
    
    def create_node(self, project_id, orig_node_id=None, parent_id = None, orig_parent=None, object_ids=None, name=None, starred = False):
        """
        Create a node.
        
        Returns:
            node ID
        """
        if parent_id is None and orig_parent is not None:
            # Subquery: Find parent by its orig_id
            parent_id = select([nodes.c.node_id], (nodes.c.orig_id == orig_parent) & (nodes.c.project_id == project_id)).as_scalar()
            # Make sure that the retrieved id is non-NULL by coalescing with -1 which will trigger an IntegrityError
            parent_id = coalesce(parent_id, -1)
            
        stmt = nodes.insert({"project_id": project_id,
                             "parent_id": parent_id,
                             "orig_id": orig_node_id,
                             "name": name,
                             "starred": starred})
        
        try:
            result = self.connection.execute(stmt)
        except SQLAlchemyError as e:
            raise TreeError("Node could not be created.") from e
        
        node_id = result.inserted_primary_key[0]
        
        if object_ids is not None:
            object_ids = iter(object_ids)
            while True:
                chunk = itertools.islice(object_ids, 1000)
                
                data = [dict(node_id=node_id, object_id = object_id, project_id = project_id) for object_id in chunk]
                
                if not data:
                    break
                
                self.connection.execute(nodes_objects.insert(), data)
        return node_id
    
    
    def _calc_centroid(self, node, children, objects):
        """
        Calculate the centroid of a node as the weighted mean of
            a) the centroids of its children, and
            b) the centroid of its own objects
        """
        # ToDo: Account for number of objects
        
        # Weight each child centroid with the number of objects it stands for
        centroids = [c["_n_objects_deep"] * c["_centroid"] for c in children if c["_centroid"] is not None]
        
        # Calculate centroid for own objects    
        object_vectors = [ o["vector"] for o in objects if o["vector"] is not None ]
        
        if len(object_vectors) > 0:
            object_centroid = np.mean(object_vectors, axis=0)
            
            # Weight object_centroid with the number of objects it stands for
            object_centroid *= node["_n_objects"]
            
            centroids.append(object_centroid)
            
        if len(centroids) > 0:
            centroid = np.mean(centroids, axis=0)
            
            try:
                # Divide by the number of objects it stands for
                centroid /= node["_n_objects_deep"]
            except TypeError:
                print("_n_objects_deep:", repr(node["_n_objects_deep"]))
                raise
            
            return centroid
        else:
            return None
        
                
    def _calc_type_objects(self, children, objects):
        """
        Calculate nine type objects for a node as
            a) a sample of nine type objects from its children, or
            b) nine of its own objects, if the node is a leaf.
        """
        if len(children) > 0:
            # Randomly subsample children
            subsample = np.random.choice(children, min(len(children), 9), replace=False)
            return [o["object_id"] 
                    for o in itertools.islice(_roundrobin([self.get_objects(c["node_id"], limit=9)
                                                           for c in subsample]), 9)]
        else:
            return [o["object_id"] for o in objects[:9]]
        
    def _calc_own_type_objects(self, children, objects):
        """
        Calculate nine own type objects for a node as
            a) the nine objects with maximum distance to the children, or
            b) [], if the node is a leaf.
        """
        
        child_vectors = np.array([ c["_centroid"] for c in children if c["_centroid"] is not None ])
        object_vectors = np.array([ o["vector"] for o in objects ])
        
        if len(child_vectors) > 0 and len(objects) > 0:
        
            try:
                classifier = Classifier(child_vectors)
                distances = classifier.distances(object_vectors)
                max_dist = np.max(distances, axis=0)
                max_dist_idx = np.argsort(max_dist)[::-1]
                
                assert len(max_dist_idx) == len(objects), "{} != {}".format(len(max_dist_idx), len(objects))
                
                return [objects[i]["object_id"] for i in max_dist_idx[:9]]
                
            except:
                print("child_vectors", child_vectors.shape)
                print("object_vectors", object_vectors.shape)
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
        rquery = select([nodes]).where(nodes.c.node_id == node["node_id"]).cte(recursive=True)
        
        parents = rquery.alias("n")
        descendants = nodes.alias("nd")
        
        rquery = rquery.union_all(
            select([descendants]).where(descendants.c.parent_id == parents.c.node_id))
        
        # For each node in rquery, calculate #objects
        deep_count = select([rquery.c.node_id, func.count(nodes_objects.c.object_id).label("count")]).\
            select_from(rquery.join(nodes_objects)).\
            group_by(rquery.c.node_id).\
            alias("deep_count")
            
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
            select([descendants]).where(descendants.c.parent_id == parents.c.node_id))
            
        # Count results
        stmt = select([func.count(rquery)])
        
        result = self.connection.scalar(stmt) or 0
        
        return result
    
    def _upgrade_node(self, node, require_valid = True, _rec_depth=0):
        """
        Parameters:
            node (dict): Fields of the node.
            require_valid: Are valid cache values required?
        """
        
        # If no valid cache values are required or the node is already valid,
        # return unchanged.
        if not require_valid or node["cache_valid"]:
            return node

        print("{}Upgrading node {}...".format(" " * _rec_depth, node["node_id"]))
        
        if node["n_children"] > 0:
            # The recursive calculation happens here
            children = self.get_children(node["node_id"], require_valid, _rec_depth = _rec_depth + 1)
        else:
            children = []
            
        # Limit to 1000 objects to speed up the calculation
        # TODO: Select objects randomly
        objects = self.get_objects(node["node_id"], limit = 1000)
        
        node["_n_objects"] = self.get_n_objects(node["node_id"])
            
        # Hint: Uses vectors
        node["_own_type_objects"] = self._calc_own_type_objects(children, objects)
        
        node["_type_objects"] = self._calc_type_objects(children, objects)
        
        node["_n_objects_deep"] = self._calc_n_objects_deep(node, children)
        
        # TODO: Remove assertion if we're sure enough that this works
        query_n_objects_deep = self._query_n_objects_deep(node)
        
        if node["_n_objects_deep"] is None:
            node["_n_objects_deep"] = query_n_objects_deep
            
        else:
            if node["_n_objects_deep"] != query_n_objects_deep:
                warnings.warn("_n_objects_deep do not match! {}!={}".format(node["_n_objects_deep"], query_n_objects_deep))
                node["_n_objects_deep"] = query_n_objects_deep
                
        # Hint: Uses vectors
        node["_centroid"] = self._calc_centroid(node, children, objects)
        if node["_centroid"] is None:
            print("Node {} has no centroid!".format(node["node_id"]))
            
        
        node["cache_valid"] = True
        
        # Store these values
        update_fields = ("cache_valid", "_centroid", "_type_objects", "_own_type_objects", "_n_objects_deep", "_n_objects")
        stmt = nodes.update().values({k: node[k] for k in update_fields}).where(nodes.c.node_id == node["node_id"])
        self.connection.execute(stmt)
                
        return node
        
        
    def get_node(self, node_id, require_valid = True):
        assert isinstance(node_id, Integral), "node_id is not integral: {!r}".format(node_id)
        
        #=======================================================================
        # stmt = text("""
        # SELECT n.*, COUNT(c.*) AS n_children
        # FROM nodes as n
        # LEFT JOIN nodes as c ON c.parent_id = n.node_id
        # GROUP BY n.node_id
        # HAVING n.node_id = :node_id""")
        #=======================================================================
        
        children = nodes.alias("children")
        
        stmt = select([nodes, func.count(children.c.node_id).label("n_children")])\
            .select_from(outerjoin(nodes, children, children.c.parent_id == nodes.c.node_id))\
            .group_by(nodes.c.node_id)\
            .having(nodes.c.node_id == node_id)
        
        result = self.connection.execute(stmt, node_id = node_id).fetchone()
        
        if result is None:
            raise TreeError("Node {} is unknown.".format(node_id))
        
        return self._upgrade_node(dict(result), require_valid)
        
        
    def get_children(self, node_id,
                     require_valid = True,
                     order_by = None,
                     include = None,
                     _rec_depth = 0):
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
        
        assert isinstance(node_id, Integral), "node_id is not integral: {!r}".format(node_id)
        
        children = nodes.alias("children")
        
        stmt = select([nodes, func.count(children.c.node_id).label("n_children")])\
            .select_from(outerjoin(nodes, children, children.c.parent_id == nodes.c.node_id))\
            .group_by(nodes.c.node_id)\
            .having(nodes.c.parent_id == node_id)
            
        if include is not None:
            stmt = stmt.having(nodes.c.starred == (include == "starred"))
            
        if order_by is not None:
            stmt = stmt.order_by(text(order_by))
            
        result = self.connection.execute(stmt, node_id = node_id).fetchall()
        
        return [self._upgrade_node(dict(r), require_valid, _rec_depth) for r in result]
        

    def merge_node_into(self, node_id, dest_node_id):
        """
        Merge a node n into another node d.
        
        All objects of n will be assigned to d.
        All children of n will be assigned to d.
        n will be deleted.
        """
        
        with self.connection.begin():
            # Get project_id of dest node
            dest_project_id = select([nodes.c.project_id]).where(nodes.c.node_id == dest_node_id).as_scalar()
            
            # Change node for objects
            stmt = nodes_objects.update().values(node_id = dest_node_id, project_id = dest_project_id).where(nodes_objects.c.node_id == node_id)
            self.connection.execute(stmt)
            
            # Change parent for children
            stmt = nodes.update().values(parent_id = dest_node_id).where(nodes.c.parent_id == node_id)
            self.connection.execute(stmt)
            
            # Delete node
            stmt = nodes.delete(nodes.c.node_id == node_id)
            self.connection.execute(stmt)
            
            # Invalidate dest node
            stmt = nodes.update().values(cache_valid = False).where(nodes.c.node_id == dest_node_id)
            self.connection.execute(stmt)
            
            
    def get_objects(self, node_id, offset = None, limit = None):
        stmt = select([objects]).select_from(objects.join(nodes_objects)).where(nodes_objects.c.node_id == node_id).offset(offset).limit(limit)
        
        result = self.connection.execute(stmt, node_id = node_id).fetchall()
        
        return [dict(r) for r in result]
    

    def get_n_objects(self, node_id):
        stmt = select([func.count()]).select_from(nodes_objects).where(nodes_objects.c.node_id == node_id)
        
        return self.connection.execute(stmt, node_id = node_id).scalar()
        
    
    def invalidate_node_and_parents(self, node_id):
        """
        Invalidate the cached values in the node and its parents.
        """
        
        stmt = text("""
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
        """)
        
        self.connection.execute(stmt, node_id = node_id)
        
        
    def recommend_children(self, node_id, max_n = 1000):
        node = self.get_node(node_id)
        
        # Get the path to the node
        path = self.get_path_ids(node_id)
        
        nodes = []
        
        # Traverse the path in reverse (without the node itself)
        for parent_id in path[:-1][::-1]:
            nodes.extend(c for c in self.get_children(parent_id) if c["node_id"] not in path)
            
            # Break if we already have enough nodes
            if len(nodes) > max_n:
                break
            
        vectors = [n["_centroid"] for n in nodes]
        
        nodes = np.array(nodes, dtype=object)
        vectors = np.array(vectors)
            
        distances = np.linalg.norm(vectors - node["_centroid"], axis=1)
        
        assert len(distances) == len(vectors), distances.shape
        
        order = np.argsort(distances)[:max_n]
        
        return nodes[order].tolist()
    
    
    def recommend_objects(self, node_id, max_n = 1000):
        node = self.get_node(node_id)
        
        # Get the path to the node (without the node itself)        
        path = self.get_path_ids(node_id)[:-1]
        
        objects = []
        
        # Traverse the parse in reverse
        for parent_id in path[::-1]:
            objects.extend(self.get_objects(parent_id))
            
            # Break if we already have enough nodes
            if len(objects) > max_n:
                break
            
        vectors = [o["vector"] for o in objects]
        
        objects = np.array(objects, dtype=object)
        vectors = np.array(vectors)
            
        distances = np.linalg.norm(vectors - node["_centroid"], axis=1)
        
        assert len(distances) == len(vectors), distances.shape
        
        order = np.argsort(distances)[:max_n]
        
        return objects[order].tolist()
    
    
    def invalidate_nodes(self, nodes_to_invalidate):
        """
        Invalidate the provided nodes.
        
        Parameters:
            nodes_to_invalidate: Collection of `node_id`s
        """
        
        stmt = nodes.update().values({nodes.c.cache_valid: False}).where(nodes.c.node_id.in_(nodes_to_invalidate))
        self.connection.execute(stmt)
    
    
    def relocate_nodes(self, node_ids, parent_id):
        """
        Relocate nodes to another parent.
        """
        
        if len(node_ids) == 0:
            return
        
        with self.connection.begin():
            
            new_parent_path = self.get_path_ids(parent_id)
        
            # Check if the new parent is below the node
            new_parent_set = set(new_parent_path)
            for node_id in node_ids:
                if node_id in new_parent_set:
                    raise TreeError("Relocating {} to {} would create a circle!".format(node_id, parent_id))
        
            #stmt = nodes.update().values({"parent_id": parent_id}).where(nodes.c.node_id.in_(node_ids))
            
            # Return distinct old `parent_id`s
            stmt = text("""
            WITH updater AS (
                UPDATE nodes x
                SET parent_id = :new_parent_id
                FROM  (SELECT node_id, parent_id FROM nodes WHERE node_id IN :node_ids FOR UPDATE) y
                WHERE  x.node_id = y.node_id
                RETURNING y.parent_id AS old_parent_id
            )
            SELECT DISTINCT old_parent_id FROM updater;
            """)
            
            # Fetch old_parent_ids
            result = self.connection.execute(stmt, new_parent_id=parent_id, node_ids=tuple(node_ids)).fetchall()

            # Invalidate subtree rooted at first common ancestor            
            parent_paths = [new_parent_path] + [self.get_path_ids(r["old_parent_id"]) for r in result]
            paths_to_update = _paths_from_common_ancestor(parent_paths)
            nodes_to_invalidate = set(sum(paths_to_update, []))
            self.invalidate_nodes(nodes_to_invalidate)
            
        
    def relocate_objects(self, object_ids, node_id):
        """
        Relocate an object to another node.
        """
        
        if len(object_ids) == 0:
            return
        
        with self.connection.begin():
            # Poject id of the new node
            project_id = select([nodes.c.project_id]).where(nodes.c.node_id == node_id)
            project_id = self.connection.execute(project_id).scalar()
            
            new_node_path = self.get_path_ids(node_id)
            
            #===================================================================
            # stmt = nodes_objects.update().values({"node_id": node_id})\
            #     .where((nodes_objects.c.object_id.in_(object_ids)) & (nodes_objects.c.project_id == project_id))
            #===================================================================
            
            # Return distinct old `parent_id`s
            stmt = text("""
            WITH updater AS (
                UPDATE nodes_objects x
                SET node_id = :node_id
                FROM  (SELECT object_id, node_id FROM nodes_objects
                    WHERE project_id = :project_id AND object_id IN :object_ids FOR UPDATE) y
                WHERE  project_id = :project_id AND x.object_id = y.object_id
                RETURNING y.node_id AS old_node_id
            )
            SELECT DISTINCT old_node_id FROM updater;
            """)
            
            # Fetch old_parent_ids
            result = self.connection.execute(stmt,
                                             node_id=node_id,
                                             project_id=project_id,
                                             object_ids=tuple(object_ids)
                                            ).fetchall()
            
            # Invalidate subtree rooted at first common ancestor
            paths = [new_node_path] + [self.get_path_ids(r["old_node_id"]) for r in result]
            paths_to_update = _paths_from_common_ancestor(paths)
            nodes_to_invalidate = set(sum(paths_to_update, []))
            self.invalidate_nodes(nodes_to_invalidate)
        
        
    #===========================================================================
    # Methods to simplify the tree
    #===========================================================================
    def flatten_tree(self, root_id):
        """
        Flatten the tree by merging nodes along the trunk.
        
        Algorithm:
            Given a node n.
            If exactly one child c of n has 2 children (grandchildren of n), merge c into n and recurse with n.
            Else recurse in all children that have 2 children by themselves (grandchildren of n).
        """
        
        
        
        print("Flattening tree...")
        
        n_total = self.node_n_descendants(root_id)
        n_processed = 0
        n_merged = 0
        
        with self.connection.begin():
            queue = [root_id]
            
            bar = ProgressBar(n_total, max_width=40)
            
            while len(queue) > 0:
                node_id = queue.pop()
                
                children = self.get_children(node_id)
            
                children_2 = [c for c in children if c["n_children"] == 2]
                
                # Count skipped children
                n_processed += len(children) - len(children_2)
                
                if len(children_2) == 0:
                    # Nothing to do for the current node
                    # Count current node
                    n_processed += 1
                    continue
                
                if len(children_2) == 1:
                    # Merge this child and restart with this node
                    child_to_merge = children_2[0]
                    self.merge_node_into(child_to_merge["node_id"], node_id)
                    n_merged += 1
                    
                    # Current node is not yet done
                    queue.append(node_id)
                    continue
                
                # Else recurse into every child that has 2 children by itself
                for c in children_2:
                    queue.append(c["node_id"])
                    
                bar.numerator = n_processed
                print(bar, end="\r")
                    
        print("Merged {:d} nodes.".format(n_merged))
                    
                    
    def prune_chains(self, root_id):
        """
        Prune chains of single nodes.
        
        Algorithm:
            Given a node n.
            If n has exactly one child c, merge this child into n and recurse with n.
            Else recurse into all children.
        """
        
        n_merged = 0
        
        with self.connection.begin():
            queue = [root_id]
            
            while len(queue) > 0:
                node_id = queue.pop()
                
                children = self.get_children(node_id)
            
                if len(children) == 0:
                    # Do nothing
                    continue
                
                if len(children) == 1:
                    # Merge this child and restart with this node
                    child_to_merge = children[0]
                    self.merge_node_into(child_to_merge["node_id"], node_id)
                    queue.append(node_id)
                    continue
                
                # Else recurse into every child
                for c in children:
                    queue.append(c["node_id"])
                    
        print("Merged {:d} nodes.".format(n_merged))

    
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
        
        stmt = text("""
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
        """)
        
        return self.connection.execute(stmt, node_id = node_id).scalar()
    
    def get_minlevel_starred(self, root_node_id, require_valid = True):
        """
        Returns all starred nodes with minimal depth.
        
        Descend into the tree while a node is not starred. Return only starred nodes.
        """
        
        rquery = select([nodes]).where(nodes.c.node_id == root_node_id).cte(recursive=True)
        
        parents = rquery.alias("n")
        descendants = nodes.alias("nd")
        
        rquery = rquery.union_all(
            # Include descendants when the parent is not starred
            select([descendants]).where((parents.c.starred == False) & (descendants.c.parent_id == parents.c.node_id)))
        
        stmt = select([rquery]).where(rquery.c.starred == True)
        
        result = self.connection.execute(stmt).fetchall()
        
        return [self._upgrade_node(dict(r), require_valid = require_valid) for r in result]

    
if __name__ in ("__main__", "builtins"):
    #: :type conn: sqlalchemy.engine.base.Connection
    with database.engine.connect() as conn:
        #=======================================================================
        # conn.execute("DROP TABLE IF EXISTS nodes_objects, nodes, projects;")
        # database.metadata.create_all(conn)
        #=======================================================================
        
        tree = Tree(conn)
        
        path = "/data1/mschroeder/NoveltyDetection/Results/CV-Clustering/2018-02-08-12-55-06/min_cluster_size-20_split-0"
        name = os.path.basename(os.path.normpath(path))
        print("Loading...")
        project_id = tree.load_project(name, path)
        root_id = tree.get_root_id(project_id)
        print(project_id, root_id)
        print("Simplifying...")
        tree.flatten_tree(root_id)
        tree.prune_chains(root_id)
        
    #===========================================================================
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
    #===========================================================================
        
        print(tree.get_projects())
    
