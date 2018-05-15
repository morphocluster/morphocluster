'''
Created on 23.02.2018

@author: mschroeder
'''

import os
import pandas as pd
import networkx as nx
import numpy as np
from _collections import deque
from morphocluster import newer_than
import joblib
from etaprogress.progress import ProgressBar
import uuid
from collections import abc
from numpy import mean
import tables

class ClusterTree(object):
    def __init__(self, path = None):
        """
        
        Parameters:
            path: Path to the project. Must contain tree.csv and objids.csv.
        """
        
        if path is not None:
            self.tree, self.root = self._load(path)
            
        self.features = None
        
    def _load(self, path):
        tree_fn = os.path.join(path, "tree.csv")
        objids_fn = os.path.join(path, "objids.csv")
        tree_cache_fn = os.path.join(path, "tree_cache.jbl")
        
        result = None
        if not newer_than((tree_fn, objids_fn), tree_cache_fn):
            try:
                result = joblib.load(tree_cache_fn)
            except Exception as e:
                print(e)
        
        if result is None:
            raw_tree = pd.read_csv(tree_fn, index_col=False)
            objids = pd.read_csv(objids_fn, index_col=False, names=["objid"], header=None, squeeze = True)
        
            result = self._build_tree(raw_tree, objids)
            
            try:
                joblib.dump(result, tree_cache_fn, protocol=-1)
            except Exception as e:
                print(e)
            
        return result
            
        
    def _build_tree(self, raw_tree, objids):
        tree = nx.DiGraph()
        
        #=======================================================================
        # cluster_mask = raw_tree["child_size"] > 1
        #=======================================================================
        root = int(raw_tree['parent'].min())
        
        for row in raw_tree.itertuples():
            if row.child_size > 1:
                tree.add_edge(row.parent, row.child)
            else:
                try:
                    tree.nodes[row.parent]["objids"].append(int(objids[row.child]))
                except KeyError:
                    tree.add_node(row.parent, objids=[int(objids[row.child])])

        return tree, root
        
    def get_children(self, parent_id):
        """
        Returns a list of children clusters for the given parent cluster.
        """
        
        return list(self.tree.successors(parent_id))
    
    def get_parent(self, node_id):
        """
        Returns the parent or None.
        """
        
        pres = self.tree.predecessors(node_id)
        
        try:
            return next(pres)
        except StopIteration:
            return None
    
    def get_out_degree(self, node_id):
        return self.tree.out_degree(node_id)
        
    def get_root(self):
        return self.root
    
    def get_leaves(self, node_id):
        return [d for d in nx.descendants(self.tree, node_id) if self.tree.out_degree(d) == 0]
    
    def get_path(self, node_id):
        return nx.shortest_path(self.tree, self.root, node_id)
    
    def get_node(self, node_id):
        return self.tree.nodes[node_id]
    
    def get_objids(self, node_ids):
        """
        Parameters:
            node_id: One or more node ids.
            
        Returns list of objids.
        """
        
        node_ids = np.ravel(node_ids)
        
        return sum((self.tree.nodes[n].get("objids", []) for n in node_ids), [])
    
    def recommend_children(self, node_id):
        return []
    
    def recommend_objects(self, node_id, max_n=None):
        """
        Recommend objects for this node.
        
        Candidates for recommendation are all objects directly below nodes
        along the path of this node (except the node itself).
        Candidates are ordered by similarity.
        """
        
        centroid = self.get_centroid(node_id)
        
        path = self.get_path(node_id)[:-1]
        
        # Empty DataFrame
        recommended_features = self.features.loc[[]]
        
        for node in path[::-1]:
            objids = self.get_objids(node)
            
            recommended_features = recommended_features.append(self.features.loc[objids])
            
            if max_n is not None and len(recommended_features) > max_n:
                break
            
        recommended_features["distance"] = np.linalg.norm(recommended_features.sub(centroid), axis=1)
        
        recommended_features.sort_values("distance")
        
        return recommended_features.index.tolist()
    
    def get_centroid(self, node_id):
        node = self.tree.nodes[node_id]
        
        try:
            centroid = node["centroid"]
        except KeyError:
            # Calculate centroid
            children = self.get_children(node_id)
            if len(children) > 0:
                centroid = np.mean([self.get_centroid(nid) for nid in children], axis=1)
                centroid /= np.linalg.norm(centroid)
            else:
                centroid = self._calc_centroid_from_objs(node_id)
                
            node["centroid"] = centroid
            
        return centroid
    
    def _calc_centroid_from_objs(self, node_id):
        objids = self.get_objids(node_id)
        centroid = self.features.loc[objids].mean(axis=0)
        
        # Scale
        centroid /= np.linalg.norm(centroid)
        
        return centroid
        
    def load_features(self, features_fn, append=True):
        print("Loading {}...".format(features_fn))
        with tables.open_file(features_fn, "r") as f_features:
            features = {t: f_features.get_node("/" + t)[:]
                       for t in ("features", "objids")}
            
            features = pd.DataFrame(features["features"], index=features["objids"])
        
        if append and self.features is not None:
            self.features = self.features.append(features)
        else:
            self.features = features
        
        print("Loaded {!r} features.".format(features.shape))
    
    def flatten_tree(self):
        """
        Recursively remove children and adopt grandchildren. 
        """
        queue = deque([self.root])
        
        while len(queue) > 0:
            node = queue.popleft()
            children = list(self.tree.successors(node))
            
            for child in children:
                grandchildren = list(self.tree.successors(child))
                
                if len(grandchildren) == 0:
                    continue
                
                child_objids = self.get_objids(child)
                
                # Remove child node
                self.tree.remove_node(child)
                
                # Append objids
                self.tree.nodes[node].setdefault("objids", []).extend(child_objids)
                
                # Link grandchildren
                self.tree.add_edges_from((node, gc) for gc in grandchildren)
                    
                # Recurse
                queue.extend(grandchildren)
                
    
    def flatten_tree2(self):
        """
        Recursively adopt leaves and only keep divisions into real branches.
        
        Prerequisite: Binary tree
        
        For a node n:
            - Merge a child if it contains a leaf (is not a real branching)
        """
        
        queue = deque([self.root])
        
        while len(queue) > 0:
            node = queue.popleft()
            
            children = list(self.tree.successors(node))
            
            for child in children:
                grandchildren = list(self.tree.successors(child))
                
                if len(grandchildren) == 0:
                    continue
                
                # Child is a real branching if none of the grandchildren are leaves
                real_branch = all(self.tree.out_degree(gc) > 0 for gc in grandchildren) 
                
                if not real_branch:
                    # Add grandchildren to node
                    child_objids = self.get_objids(child)
                    
                    # Remove child node
                    self.tree.remove_node(child)
                    
                    # Append objids
                    self.tree.nodes[node].setdefault("objids", []).extend(child_objids)
                    
                    # Link grandchildren
                    self.tree.add_edges_from((node, gc) for gc in grandchildren)
                    
                    # Recurse
                    queue.extend(grandchildren)
                else:
                    # Recurse into child
                    queue.append(child)
                    
    def flatten_tree3(self):
        """
        Recursively adopt the trunk grandchild as sibling of the other children.
        
        Prerequisite: Binary tree
        
        For a node n:
            - Merge a child if it contains a leaf (is not a real branching)
        """
        
        queue = deque([self.root])
        
        while len(queue) > 0:
            node = queue.popleft()
            
            children = list(self.tree.successors(node))
            
            for child in children:
                grandchildren = list(self.tree.successors(child))
                
                if len(grandchildren) == 0:
                    continue
                
                n_child_descendants = len(nx.descendants(self.tree, child))
                
                n_grandchildren_descendants = [len(nx.descendants(self.tree, gc)) for gc in grandchildren]
                
                trunk_idx = np.argmax(n_grandchildren_descendants)
                
                if n_grandchildren_descendants[trunk_idx] > n_child_descendants * 0.75:
                    # If the largest grandchild has at least 75% of all descendants
                    # adopt it as a child
                    
                    trunk_grandchild = grandchildren[trunk_idx]
                    
                    # Remove grandchild from current child
                    self.tree.remove_edge(child, trunk_grandchild)
                    
                    # Add grandchild to current node
                    self.tree.add_edge(node, trunk_grandchild)
                    
                    # Recurse into trunk
                    queue.append(trunk_grandchild)
                else:
                    # Recurse into child
                    queue.append(child)
                    
    def flatten_tree4(self, p=0.75):
        """
        Recursively adopt the trunk grandchild as child of the current trunk.
        
        If there is no significant trunk grandchild, recurse with current child as trunk.
        
        """
        
        queue = deque([(self.root, self.root)])
        
        n_nodes_total = len(self.tree.nodes)
        
        bar = ProgressBar(n_nodes_total, max_width=40)
        
        while len(queue) > 0:
            node, trunk = queue.popleft()
            
            children = list(self.tree.successors(node))
            
            for child in children:
                grandchildren = list(self.tree.successors(child))
                
                if len(grandchildren) == 0:
                    continue
                
                n_gc_descendants = [len(nx.descendants(self.tree, gc)) for gc in grandchildren]
                
                n_gc_descendants_total = sum(n_gc_descendants)
                
                trunk_idx = np.argmax(n_gc_descendants)
                
                if n_gc_descendants[trunk_idx] > n_gc_descendants_total * p:
                    # If the largest grandchild has at p of all descendants, it is the continuation of the trunk.
                    # Adopt it as a child of the current trunk.
                    
                    trunk_grandchild = grandchildren[trunk_idx]
                    
                    # Remove grandchild from current child
                    self.tree.remove_edge(child, trunk_grandchild)
                    
                    # Add grandchild to current trunk
                    self.tree.add_edge(trunk, trunk_grandchild)
                    
                    # Recurse into trunk grandchild with trunk
                    queue.append((trunk_grandchild, trunk))
                    
                    # Recurse into child with trunk
                    queue.append((child, trunk))
                else:
                    # Recurse into child and make the child the new trunk
                    queue.append((child, child))
            
            bar.numerator += 1
            print(bar, end="\r")
        print()
                    
    def prune_single_children(self):
        """
        Collapse nodes that only have one child.
        """
        
        queue = deque([self.root])
        
        while len(queue) > 0:
            root = queue.popleft()
            
            children = list(self.tree.successors(root))
            
            for child in children:
                grandchildren = list(self.tree.successors(child))
                
                if len(grandchildren) == 1:
                    # If child has only one child (grandchild of root),
                    # remove child and attach the objids and the grandchild to root.
                
                    child_objids = self.tree.nodes[child].get("objids", [])
                    root_objids = self.tree.nodes[child].setdefault("objids", [])
                    root_objids.extend(child_objids)
                
                    self.tree.remove_node(child)
                    
                    for gc in grandchildren:
                        self.tree.add_edge(root, gc)
                    
                    # Visit root again to collapse longer chains of single children
                    queue.append(root)
                else:
                    # Recurse into child
                    queue.append(child)
                    
    def print_stats(self):
        print("Number of leaves:", len(self.get_leaves(self.root)))
        print("Number of toplevel children:", len(self.get_children(self.root)))
        
    def create_node(self, parent, children=[], data = {}):
        id_ = self.get_next_id()
        
        if not self.tree.has_node(parent):
            raise ValueError("Parent {} is not in the tree.".format(repr(parent)))
        
        self.tree.add_node(id_, **data)
        self.tree.add_edge(parent, id_)
        
        if len(children):
            for child in children:
                pres = list(self.tree.predecessors(child))
                for pre in pres:
                    self.tree.remove_edge(pre, child)
                self.tree.add_edge(id_, child)
                
        return id_
        
    def get_next_id(self):
        return max(self.tree.nodes) + 1
            
if __name__ in ("__main__", "builtins"):
    from morphocluster.cluster_tree import ClusterTree
    print("Loading...")
    tree = ClusterTree("/data1/mschroeder/NoveltyDetection/Results/CV-Clustering/2018-02-08-12-55-06/min_cluster_size-30_split-1")
    tree.load_features("/data1/mschroeder/NoveltyDetection/Results/CrossVal/2018-01-26-11-54-41/n_features-32_split-0/collection_unlabeled.h5")
    
    