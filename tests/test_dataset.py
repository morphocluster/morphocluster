import os.path

import pandas as pd
import pytest
from sqlalchemy import func, select

from morphocluster.dataset import Dataset
from morphocluster.extensions import database
from morphocluster.models import nodes
from morphocluster.processing import Tree
from morphocluster.project import Project, ProjectError


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
    with pytest.raises(ProjectError):
        with project:
            project.import_tree(datadir / "tree.zip")

    orig_tree = Tree.from_saved(str(datadir / "tree.zip"))

    # Assert that get_objects retrieves the right objects
    with project:
        for node_id in orig_tree.nodes["node_id"]:
            db_objects = project.get_objects(node_id)

            orig_object_selector = orig_tree.objects["node_id"] == node_id
            orig_object_ids = orig_tree.objects.loc[orig_object_selector, "object_id"]

            db_object_ids = {o["object_id"] for o in db_objects}

            assert (
                set(orig_object_ids) == db_object_ids
            ), "Objects do not match for node_id={}".format(node_id)

    # Assert that consolidate_node works as expected
    with project:
        root_id = project.get_root_id()
        result = project.consolidate_node(root_id, return_="raw")
        # TODO

    # Assert exported tree is the same as imported
    with project:
        db_tree = project.export_tree()

        node_columns = sorted(orig_tree.nodes.columns)
        orig_nodes = (
            orig_tree.nodes[node_columns]
            .sort_values(by="node_id")
            .reset_index(drop=True)
        )
        db_nodes = (
            db_tree.nodes[node_columns].sort_values(by="node_id").reset_index(drop=True)
        )

        for name in node_columns:
            assert list(null2None(orig_nodes[name])) == list(null2None(db_nodes[name]))

        object_columns = sorted(orig_tree.objects.columns)
        orig_objects = (
            orig_tree.objects[object_columns]
            .sort_values(by="object_id")
            .reset_index(drop=True)
        )
        db_objects = (
            db_tree.objects[object_columns]
            .sort_values(by="object_id")
            .reset_index(drop=True)
        )

        for name in object_columns:
            assert list(null2None(orig_objects[name])) == list(
                null2None(db_objects[name])
            )

    # Assert create_node can calculate node_id
    with project:
        node_id = project.create_node()
        assert node_id is not None


def null2None(df):
    return df.where((pd.notnull(df)), None)
