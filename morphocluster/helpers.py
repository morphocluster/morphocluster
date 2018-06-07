'''
Created on 06.06.2018

@author: mschroeder
'''

import numpy as np
from itertools import chain

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