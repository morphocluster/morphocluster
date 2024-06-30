"""
JSON encoder that is able to serialize numpy scalars.
"""
from flask.json import JSONEncoder
import numpy as np


class NumpyJSONEncoder(JSONEncoder):
    """
    JSON encoder that is able to serialize numpy scalars.
    """

    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return JSONEncoder.default(self, o)
