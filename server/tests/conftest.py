import multiprocessing
import os.path
import pathlib
import signal

import flask_migrate
import psycopg2
import pytest
import redis
import sqlalchemy.exc

from morphocluster.server import create_app
from morphocluster.server.cli import _add_user
from morphocluster.server.extensions import database
from lovely.pytest.docker.compose import Services


def check_postgres(ip, port):
    try:
        conn = psycopg2.connect(
            host=ip, port=port, user="morphocluster", password="morphocluster"
        )
        conn.close()
        return True
    except psycopg2.OperationalError as exc:
        return False


def check_redis(ip, port):
    try:
        r = redis.Redis(host=ip, port=port)
        r.info()
        return True
    except redis.ConnectionError:
        return False


@pytest.fixture(scope="session")
def docker_postgres(docker_services: Services):
    docker_services.start("postgres")
    public_port = docker_services.wait_for_service(
        "postgres", 5432, check_server=check_postgres
    )
    url = "postgresql://morphocluster:morphocluster@{docker_services.docker_ip}:{public_port}/morphocluster".format(
        docker_services=docker_services, public_port=public_port
    )
    return url


@pytest.fixture(scope="session")
def docker_redis_persistent(docker_services: Services):
    docker_services.start("redis-persistent")
    public_port = docker_services.wait_for_service(
        "redis-persistent", 6379, check_server=check_redis
    )
    url = "redis://{docker_services.docker_ip}:{public_port}/0".format(
        docker_services=docker_services, public_port=public_port
    )
    return url


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("session")


@pytest.fixture(scope="session")
def flask_app(docker_postgres, docker_redis_persistent, session_tmp_path: pathlib.Path):
    """Create and configure a new app instance for the session."""

    data_dir = session_tmp_path / "data"
    data_dir.mkdir()

    # create the app with common test config
    app = create_app(
        {
            "SQLALCHEMY_DATABASE_URI": docker_postgres,
            "RQ_REDIS_URL": docker_redis_persistent,
            "DATA_DIR": str(data_dir),
        }
    )

    with app.app_context():
        flask_migrate.upgrade()

        try:
            with database.engine.begin():
                _add_user("test_user", "test_user")
        except sqlalchemy.exc.IntegrityError:
            # The user exists already
            pass

        yield app

    # Cleanup


@pytest.fixture(scope="session")
def flask_rq_worker(flask_app):

    ctx = multiprocessing.get_context("fork")

    print("Starting worker...")
    runner = flask_app.test_cli_runner()
    p = ctx.Process(target=lambda: runner.invoke(args=["rq", "worker"]))
    p.start()

    yield

    print("Stopping worker...")
    os.kill(p.pid, signal.SIGINT)
    p.join(1.0)

    print("Worker stopped.")


@pytest.fixture(scope="session")
def flask_client(flask_app):
    with flask_app.test_client() as c:
        yield c


@pytest.fixture(scope="session")
def flask_cli(flask_app):
    with flask_app.test_cli_runner() as r:
        yield r


@pytest.fixture(scope="session")
def datadir():
    return pathlib.Path(__file__).parent.parent / "data"
