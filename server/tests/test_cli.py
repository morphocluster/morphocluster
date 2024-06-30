import re
import uuid

from flask.app import Flask


def test_load(flask_app, datadir):
    runner = flask_app.test_cli_runner()

    # Load objects
    result = runner.invoke(
        args=[
            "load-objects",
            str(datadir / "example" / "objects.zip"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    match = re.search(r"Done\.", result.output)
    assert match is not None, result.output

    # Load features
    result = runner.invoke(
        args=[
            "load-features",
            str(datadir / "example" / "features.h5"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    match = re.search(r"Done\.", result.output)
    assert match is not None, result.output

    # Load project
    result = runner.invoke(
        args=[
            "load-project",
            str(datadir / "example" / "tree.zip"),
            "test_project",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    match = re.search(r"Project ID: (.+)", result.output)
    assert match is not None, result.output

def test_user(flask_app: Flask):
    runner = flask_app.test_cli_runner()

    username = f"user_{uuid.uuid4().hex}"

    # Create user
    result = runner.invoke(
        args=[
            "add-user",
            username
        ],
        catch_exceptions=False,
        input="password\npassword\n"
    )
    assert result.exit_code == 0, result.output

    match = re.search(re.escape(f"User {username} added."), result.output)
    assert match is not None, result.output

    # Change user
    result = runner.invoke(
        args=[
            "change-user",
            username
        ],
        catch_exceptions=False,
        input="password2\npassword2\n"
    )
    assert result.exit_code == 0, result.output

    match = re.search(re.escape(f"User {username} changed."), result.output)
    assert match is not None, result.output