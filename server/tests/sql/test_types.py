import pytest
from sqlalchemy import Column, Table

from morphocluster.server.extensions import database
from morphocluster.server.sql.types import Point

_point_table = Table(
    "point_table",
    database.metadata,
    Column("point", Point),
)


@pytest.fixture(scope="module")
def db_connection(flask_app):
    with database.engine.begin() as connection:
        yield connection


@pytest.fixture(scope="module")
def point_table(flask_app, db_connection):
    with db_connection.begin() as txn:
        _point_table.create(db_connection)

        yield _point_table

        _point_table.drop(db_connection)

        txn.rollback()


# @pytest.mark.skip()
def test_point(point_table, db_connection):
    values_target = [
        # Points
        ((1, 2), (1.0, 2.0)),
        # NULL
        (None, None),
    ]
    db_connection.execute(
        point_table.insert(),
        [dict(point=v[0]) for v in values_target],
    )

    values_actual = [
        r[0] for r in db_connection.execute(point_table.select()).fetchall()
    ]

    assert values_actual == [v[1] for v in values_target]
