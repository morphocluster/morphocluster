import os.path

import flask_migrate
import psycopg2
import pytest

from morphocluster import create_app
from morphocluster.cli import _add_user
from morphocluster.extensions import database
import pathlib


def check_postgres(ip, port):
    try:
        conn = psycopg2.connect(
            host=ip, port=port, user="morphocluster", password="morphocluster"
        )
        conn.close()
        return True
    except psycopg2.OperationalError as exc:
        return False


@pytest.fixture(scope='session')
def docker_postgres(docker_services):
    docker_services.start('postgres')
    public_port = docker_services.wait_for_service(
        "postgres", 5432, check_server=check_postgres
    )
    url = "postgresql://morphocluster:morphocluster@{docker_services.docker_ip}:{public_port}/morphocluster".format(
        docker_services=docker_services,
        public_port=public_port,
    )
    return url


@pytest.fixture(scope='session')
def session_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("session")


@pytest.fixture(scope='session')
def flask_app(docker_postgres, session_tmp_path: pathlib.Path):
    """Create and configure a new app instance for the session."""

    data_dir = session_tmp_path / "data"
    data_dir.mkdir()

    # create the app with common test config
    app = create_app(
        {
            "SQLALCHEMY_DATABASE_URI": docker_postgres,
            "DATA_DIR": str(data_dir),
        }
    )

    with app.app_context():
        flask_migrate.upgrade()

        _add_user("test", "test")

        yield app

    # Cleanup


@pytest.fixture(scope='session')
def flask_client(flask_app):
    with flask_app.test_client() as c:
        yield c


@pytest.fixture(scope='session')
def flask_cli(flask_app):
    with flask_app.test_cli_runner() as r:
        yield r


@pytest.fixture(scope='session')
def datadir():
    return pathlib.Path(__file__).parent / "data"