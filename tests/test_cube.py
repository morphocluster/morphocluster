import pytest
from sqlalchemy import Column, Table

from morphocluster.extensions import database as db
from morphocluster.cube import Cube, parse

metadata = db.metadata

_cube_table = Table(
    'cube_table',
    metadata,
    Column("cube", Cube),
)


@pytest.fixture(scope="session")
def cube_table(flask_app):
    connection = db.get_connection()
    with connection.begin() as txn:
        _cube_table.create(connection)
        yield _cube_table
        _cube_table.drop(connection)

        txn.rollback()


# @pytest.mark.skip()
def test_cube(cube_table):
    connection = db.get_connection()

    values_target = [
        # Number: 1d point
        (1.0, (1.0, )),
        (float("inf"), (float("inf"), )),
        # (float("nan"), (float("nan"), )), # Problem: nan != nan

        # Points
        ((), ()),
        ((1, 2), (1., 2.)),

        # Cubes
        ([(1, 2), (3, 4)], [(1.0, 2.0), (3.0, 4.0)]),

        (None, None),
    ]
    connection.execute(
        cube_table.insert(),
        [dict(cube=v[0]) for v in values_target],
    )

    values_actual = [
        r[0] for r in connection.execute(cube_table.select()).fetchall()
    ]

    assert values_actual == [v[1] for v in values_target]


def test_parse():
    values = [
        ("()", ()),
        ('(1)', (1.0, )),
        ('(Infinity)', (float("inf"), )),
        ('(1,2),(3,4)', [(1.0, 2.0), (3.0, 4.0)]),
    ]

    results = [parse(v[0]) for v in values]

    assert results == [v[1] for v in values]
