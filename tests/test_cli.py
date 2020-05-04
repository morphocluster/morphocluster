import re


def test_load(flask_app, datadir):
    runner = flask_app.test_cli_runner()

    # Create dataset with objects and features
    result = runner.invoke(
        args=[
            "dataset",
            "create",
            "test_dataset",
            "test_user",
            "--objects",
            str(datadir / "objects.zip"),
            "--features",
            str(datadir / "features.h5"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    match = re.search(r"Created dataset: (.+) \(id (\d+)\) for (.+)\.", result.output)
    assert match is not None, result.output

    dataset_id = int(match[2])

    # Load project
    result = runner.invoke(
        args=[
            "project",
            "create",
            "test_project",
            str(dataset_id),
            "--tree",
            str(datadir / "tree.zip"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    match = re.search(r"Created project: (.+) \(id (\d+)\) in (.+)\.", result.output)
    assert match is not None, result.output
