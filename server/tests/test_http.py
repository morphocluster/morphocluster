from requests.auth import _basic_auth_str


def test_auth(flask_client):
    # A request without authorization should fail with 401
    response = flask_client.get("/")
    assert response.status_code == 401

    # A request with authorization should not fail with 401
    headers = {"Authorization": _basic_auth_str("test_user", "test_user")}
    response = flask_client.get("/", headers=headers)
    assert response.status_code != 401
