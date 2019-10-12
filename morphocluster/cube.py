import numbers

from sqlalchemy.types import UserDefinedType
import numpy as np


def parse(string: str):
    result = [
        tuple(float(x) for x in part.strip("()").split(",") if x)
        for part in string.split("),(")
    ]

    if len(result) == 1:
        return result[0]

    return result


class Cube(UserDefinedType):
    def __init__(self, as_numpy=False):
        # TODO: as_numpy
        self.as_numpy = as_numpy

    def get_col_spec(self, **kw):
        return "cube"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None

            if isinstance(value, (numbers.Number)):
                return str(value)

            if isinstance(value, tuple):
                return "(" + ",".join(map(process, value)) + ")"

            if isinstance(value, list):
                return "[" + ",".join(map(process, value)) + "]"

            if isinstance(value, np.ndarray):
                ndim = value.ndim
                if ndim == 1:
                    return process(tuple(value))
                if ndim == 2:
                    return process([tuple(v) for v in value])

                raise ValueError("Unexpected number of dimensions: {}".format(ndim))

            raise ValueError("Unexpected value({}): {!r}".format(type(value), value))

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return value
            return parse(value)

        return process
