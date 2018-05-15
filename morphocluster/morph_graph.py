'''
Created on 28.02.2018

@author: mschroeder
'''

import pandas as pd
import networkx as nx
import os
from base64 import urlsafe_b64encode
import time
from itertools import islice

class MorphGraph():
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def import_clusters(self, fname):
        cluster_labels = pd.read_csv(fname, index_col=False)
        
        unique_prefix = str(time.time()) + "."
        
        for row in cluster_labels.itertuples():
            cluster_id = unique_prefix + str(row.label)
            
            self.graph.add_node(row.objid, is_obj=True)
            self.graph.add_edge(cluster_id, row.objid)
            
if __name__ in ("__main__", "builtins"):
    g = MorphGraph()
    
    g.import_clusters("/data1/mschroeder/NoveltyDetection/Results/CV-Clustering/2018-02-08-12-55-06/min_cluster_size-30_split-1/cluster_labels.csv")
    
    for node in islice(g.graph.nodes, 100):
        print(node, g.graph.nodes[node])