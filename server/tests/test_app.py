import pytest
from requests.auth import _basic_auth_str

from morphocluster.server.extensions import database
from requests.auth import _basic_auth_str


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
    headers = {"Authorization": _basic_auth_str("test_user", "test_user")}
    response = flask_client.get("/", headers=headers)
    assert response.status_code != 401
