'''
Created on 06.06.2018

@author: mschroeder
'''

import numpy as np
from itertools import chain
from collections import defaultdict

def seq2array(seq, length):
    """
    Converts a sequence consisting of `numpy array`s to a single array.
    Elements that are None are converted to an appropriate zero entry.
    """
    
    seq = iter(seq)
    leading = []
    zero = None
    
    for x in seq:
        leading.append(x)
        if x is not None:
            zero = np.zeros_like(x)
            break
        
    if zero is None:
        raise ValueError("Empty sequence or only None")
    
    array = np.empty((length,) + zero.shape, zero.dtype)
    for i, x in enumerate(chain(leading, seq)):
        array[i] = zero if x is None else x 
    
    return array

class keydefaultdict(defaultdict):
    """
    The first argument provides the initial value for the default_factory attribute.
    
    Attributes:
        default_factory: Callable that produces a missing value given its key.
    
    By Jochen Ritzel
    https://stackoverflow.com/a/2912455/1116842
    """
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret