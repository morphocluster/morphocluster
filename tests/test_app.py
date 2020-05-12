import os
import tempfile

import pytest
from requests.auth import _basic_auth_str

from morphocluster import create_app
from morphocluster.extensions import database
from morphocluster.cli import _add_user
from requests.auth import _basic_auth_str


@pytest.fixture(scope="session")
def flask_app():
    """Create and configure a new app instance for each test."""
    # create the app with common test config
    app = create_app(
        {
            "SQLALCHEMY_DATABASE_URI": "postgresql://morphocluster_test:morphocluster_test@localhost/morphocluster_test",
        }
    )

    with app.app_context():
        # Initialize db
        with database.engine.begin() as txn:
            database.metadata.drop_all(txn)
            database.metadata.create_all(txn)

        _add_user("test", "test")

        yield app

    # Cleanup


@pytest.fixture(scope="session")
def flask_client(flask_app):
    with flask_app.test_client() as c:
        yield c


@pytest.fixture(scope="session")
def flask_cli(flask_app):
    with flask_app.test_cli_runner() as r:
        yield r


def test_auth(flask_client):
    # A request without authorization should fail with 401
    response = flask_client.get("/")
    assert response.status_code == 401

    # A request with authorization should not fail with 401
    headers = {"Authorization": _basic_auth_str("test", "test")}
    response = flask_client.get("/", headers=headers)
    assert response.status_code != 401
