from sqlalchemy.types import TypeEngine, UserDefinedType, Float


class Point(UserDefinedType):
    """
    Represent a point by using a zero-volume cube from the cube extension.

    This allows to calculate distances inside the database.

    https://www.postgresql.org/docs/current/cube.html
    """

    def __init__(self, numpy=False) -> None:
        self.numpy = numpy

        super().__init__()

    def get_col_spec(self, **kw):
        return "CUBE"

    def bind_processor(self, dialect):
        def process(value):
            # NULL
            if value is None:
                return None

            return "(" + ",".join(str(v) for v in value) + ")"

        return process

    def result_processor(self, dialect, coltype):

        if self.numpy:
            import numpy as np

            def process_numpy(value):
                # NULL
                if value is None:
                    return value

                if isinstance(value, memoryview):
                    value = value.tobytes()

                return np.fromstring(value[1:-1], sep=",")

            return process_numpy

        def process_tuple(value):
            # NULL
            if value is None:
                return value

            if isinstance(value, memoryview):
                value = value.tobytes()
                sep = b","
            else:
                sep = ","

            return tuple(float(v) for v in value[1:-1].split(sep))

        return process_tuple

    class Comparator(TypeEngine.Comparator):
        def dist_euclidean(self, other):
            return self.op("<->", return_type=Float)(other)

    comparator_factory = Comparator
    
    # Statements using this type are safe to cache.
    cache_ok = True
