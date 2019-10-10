import os.path

import pytest
from sqlalchemy import func, select

from morphocluster.dataset import Dataset
from morphocluster.extensions import database
from morphocluster.models import nodes
from morphocluster.processing import Tree
from morphocluster.project import Project


@pytest.fixture()
def dataset(flask_app, datadir):
    dataset = Dataset.create("test_dataset")

    dataset.load_objects(datadir / "objects.zip")
    dataset.load_object_features(datadir / "features.h5")

    yield dataset

    dataset_path = dataset.path
    dataset.remove()

    assert not os.path.isdir(dataset_path)


@pytest.fixture()
def project(dataset: Dataset, datadir):
    project: Project = dataset.create_project("test_project")

    connection = database.get_connection()

    with project:
        project.import_tree(datadir / "tree.zip")

        # Assert that a root ID exists at this point
        project.get_root_id()

    yield project

    with project:
        project.remove()


def test_project(project: Project, datadir):
    # Assert project is listed
    projects = Project.get_all()
    assert len(projects) == 1

    # Test reentrancy
    with project:
        with project:
            pass

    # Assert project may not import a second tree
    with pytest.raises(RuntimeError):
        with project:
            project.import_tree(datadir / "tree.zip")

    # # Assert exported tree is the same as imported
    # with project:
    #     db_tree = project.export_tree()
    #     orig_tree = Tree.from_saved(str(datadir / "tree.zip"))

    #     assert set(db_tree.nodes.itertuples()) == set(
    #         orig_tree.nodes.itertuples()
    #     )

    #     assert set(db_tree.objects.itertuples()) == set(
    #         orig_tree.objects.itertuples()
    #     )

    # Assert create_node can calculate node_id
    with project:
        node_id = project.create_node()
