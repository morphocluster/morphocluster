"""
Created on 01.03.2018

@author: mschroeder
"""
import os


def newer_than(a, b):
    """
    Returns true if file(s) in a are newer than file(s) in b or file(s) in b do not exist.
    """

    if isinstance(a, str):
        a = [a]

    if isinstance(b, str):
        b = [b]

    # Find the latest modification of files in a
    a_time = max(os.path.getctime(fn) for fn in a)

    # Find the earliest modification of files in b
    b_time = min((os.path.getctime(fn) for fn in b if os.path.exists(fn)), default=0)

    return a_time > b_time
