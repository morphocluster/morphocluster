'''
Created on 10.08.2018

@author: mschroeder
'''

import numpy as np
from collections.abc import Sequence
from morphocluster.helpers import seq2array

class MemberCollection(Sequence):
    """
    A collection of nodes or objects or both.
    
    Nodes have a _centroid key, objects have a _ 
    
    Parameters:
        members: list of dict
        none_action: "raise" | "zero" | "remove"
            How to deal with None vector values.
    """
    
    def __init__(self, members, none_action="raise"):
        self.members = members
        self.none_action = none_action
        
    @property
    def vectors(self):
        try:
            return self._vectors
        except AttributeError:
            vectors = [ m["_centroid"] if "_centroid" in m else m["vector"] for m in self.members ]
            
            if self.none_action == "raise":
                self._vectors = np.array(vectors)
            elif self.none_action == "zero":
                self._vectors = seq2array(vectors, len(vectors))
            elif self.none_action == "remove":
                self._vectors = np.array([v for v in vectors if v is not None])
            else:
                NotImplementedError(self.none_action)
                
            return self._vectors
        
    # Abstract methods
    def __getitem__(self, index):
        return self.members[index]

    def __len__(self):
        return len(self.members)
