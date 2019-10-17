import os.path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal
from timer_cm import Timer

from morphocluster.dataset import Dataset
from morphocluster.extensions import database
from morphocluster.helpers import seq2array
from morphocluster.processing import Tree
from morphocluster.project import Project, ProjectError, commonprefix


def null2None(df):
    return df.where((pd.notnull(df)), None)


@pytest.fixture(name="dataset", scope="session")
def _dataset(flask_app, datadir):
    dataset = Dataset.create("test_dataset")

    dataset.load_objects(datadir / "objects.zip")
    dataset.load_object_features(datadir / "features.h5")

    yield dataset

    dataset_path = dataset.path
    dataset.remove()

    assert not os.path.isdir(dataset_path)


@pytest.fixture(name="project", scope="session")
def _project(dataset: Dataset, datadir):
    project: Project = dataset.create_project("test_project")

    connection = database.get_connection()

    with project:
        project.import_tree(datadir / "tree.zip")

        # Assert that a root ID exists at this point
        project.get_root_id()

    # Assert project is listed
    projects = Project.get_all()
    assert len(projects) == 1

    yield project

    with project:
        project.remove()


@pytest.fixture(name="orig_tree", scope="session")
def _orig_tree(datadir):
    return Tree.from_saved(str(datadir / "tree.zip"))


def test_consolidate_node_raw(project, orig_tree, datadir):
    # Assert that consolidate_node works as expected
    features_fn = str(datadir / "features.h5")
    with project, h5py.File(features_fn, "r") as f_features:
        object_ids = f_features["object_id"]
        vectors = f_features["features"]

        root_id = project.get_root_id()
        result = project.consolidate_node(
            root_id, depth=-1, return_="raw", exact_vector="exact"
        )

        for node_id, values in result.iterrows():
            ## Assert that n_children_ is correct
            n_children_orig = (orig_tree.nodes["parent_id"] == node_id).sum()
            assert n_children_orig == values["n_children_"]

            ## Assert that n_objects_own_ is correct
            n_objects_own_orig = (orig_tree.objects["node_id"] == node_id).sum()
            assert n_objects_own_orig == values["n_objects_own_"]

            ## Assert that vector_own_ is correct
            object_selector = orig_tree.objects["node_id"] == node_id
            node_object_ids = orig_tree.objects.loc[object_selector, "object_id"]
            h5_selector = pd.Series(object_ids).isin(node_object_ids)

            if h5_selector.any():
                vector_own_h5 = vectors[h5_selector].sum(axis=0)
                assert_allclose(
                    vector_own_h5,
                    values["vector_own_"],
                    atol=1e-6,
                    err_msg="Unexpected vector_own_ for node_id={}".format(node_id),
                )
            else:
                assert values["vector_own_"] == None

            # TODO: Other cached values


def test_match_imported(project, orig_tree):
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


def test_reentrancy(project: Project):
    with project:
        with project:
            pass

    # TODO: Exception


def test_no_second_import(project: Project, datadir):
    # Assert project may not import a second tree
    with pytest.raises(ProjectError):
        with project:
            project.import_tree(datadir / "tree.zip")


def test_get_objects(project: Project, orig_tree: pd.DataFrame):
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


def test_create_node(project: Project):
    # Assert create_node can calculate node_id
    with project:
        node_id = project.create_node()
        assert node_id is not None


def assert_vectors_equal(node_ids, vec_a, vec_b, atol=1e-5):
    __tracebackhide__ = True

    dist = np.linalg.norm(vec_a - vec_b, axis=1)
    assert len(dist) == len(node_ids)
    dist_ok = dist <= atol
    p_match = np.mean(dist_ok)
    n_match = np.sum(dist_ok)
    n = len(dist_ok)
    n_miss = n - n_match
    assert dist_ok.all(), "Matches {:.3%} ({:d}+{:d}={:d}). Distances: {}".format(
        p_match,
        n_match,
        n_miss,
        n,
        [{nid: d} for nid, d in zip(node_ids[~dist_ok], dist[~dist_ok])],
    )


def test_commonprefix():
    assert commonprefix([[1, 2, 3], [1, 2, 4, 5], [1, 2, 6, 7, 8]]) == [1, 2]
    assert commonprefix([[1, 2, 3], [1, 4, 5], [1, 6, 7, 8]]) == [1]
    assert commonprefix([[1], [1, 2, 3], [1, 4, 5], [1, 6, 7, 8]]) == []


def test_relocate_objects(project: Project, orig_tree: pd.DataFrame):
    # Assert relocate_objects recalculates the vectors
    with database.get_connection().begin() as txn:
        with project:
            root_id = project.get_root_id()
            n_objects = project.get_n_objects()

            # Ignore these columns in comparisons because they are indeterministic or insignificant
            IGNORE_COLUMNS = ["type_objects_own_", "type_objects_", "cache_valid"]

            # Make sure tree is up to date and record tree before relocate
            project.reset_cached_values()
            tree0 = project.consolidate_node(
                root_id, depth=-1, exact_vector="exact", return_="raw"
            )
            tree0, _ = _prepare_tree(tree0, ["vector_own_", "vector_"], IGNORE_COLUMNS)

            assert tree0.loc[root_id, "n_objects_"] == n_objects

            # Sample some object_ids
            objects_sample = orig_tree.objects.sample(100)
            object_ids = objects_sample["object_id"]
            old_node_ids = objects_sample["node_id"].unique()
            print("{} old node_ids.".format(len(old_node_ids)))
            old_paths = [project.get_path(int(nid)) for nid in old_node_ids]
            affected_precursers = set(sum(old_paths, []))

            # Select some node_id
            dest_node_id = int(orig_tree.nodes["node_id"].sample().squeeze())
            dest_node_path = project.get_path(dest_node_id)
            affected_precursers.update(dest_node_path)

            prefix = commonprefix(old_paths + [dest_node_path])
            len_prefix = len(prefix)
            affected_subtree_nodes = set(sum((p[len_prefix:] for p in old_paths), []))
            affected_subtree_nodes.update(dest_node_path[len_prefix:])

            affected_precursers = sorted(affected_precursers)

            print("{} affected precursors.".format(len(affected_precursers)))
            print("{} affected subtree nodes.".format(len(affected_subtree_nodes)))

            ## Relocate objects
            project.relocate_objects(object_ids, dest_node_id)

            # These properties can change between before and after relocate
            FLAG_SUMMARY_NAMES = [
                "n_approved_objects_",
                "n_filled_objects_",
                "n_preferred_objects_",
            ]

            # Record tree after relocate
            tree1 = project.get_subtree(root_id)
            tree1, (vector_own_1, vector_1) = _prepare_tree(
                tree1, ["vector_own_", "vector_"], IGNORE_COLUMNS
            )

            # relocate_objects has to reset all flag summaries for all predecessors
            # in
            for flag_summary_name in FLAG_SUMMARY_NAMES:
                assert pd.isna(
                    tree1.loc[affected_precursers, flag_summary_name]
                ).all(), "{} has to be reset for all predecessors".format(
                    flag_summary_name
                )

            # Numbers for root must not change
            # print(tree1.drop(columns=FLAG_SUMMARY_NAMES).columns)
            assert_series_equal(
                tree1.drop(columns=FLAG_SUMMARY_NAMES).loc[root_id],
                tree0.drop(columns=FLAG_SUMMARY_NAMES).loc[root_id],
            )

            # consolidate to fill in the holes left by update_cached_values
            with Timer("consolidate_partial") as t:
                tree2 = project.consolidate_node(
                    root_id, depth=-1, exact_vector="exact", return_="raw"
                )
            time_consolidate_partial = t.elapsed

            tree2, (vector_own_2, vector_2) = _prepare_tree(
                tree2, ["vector_own_", "vector_"], IGNORE_COLUMNS
            )

            # Numbers for root must not change
            assert_series_equal(
                tree2.drop(columns=FLAG_SUMMARY_NAMES).loc[root_id],
                tree0.drop(columns=FLAG_SUMMARY_NAMES).loc[root_id],
            )

            # reset and consolidate again to compare to fresh values
            project.reset_cached_values()
            with Timer("consolidate_full") as t:
                tree3 = project.consolidate_node(
                    root_id, depth=-1, exact_vector="exact", return_="raw"
                )
            time_consolidate_full = t.elapsed
            tree3, (vector_own_3, vector_3) = _prepare_tree(
                tree3, ["vector_own_", "vector_"], IGNORE_COLUMNS
            )

            # TODO: Assert that partial consolidate is faster
            # assert time_consolidate_partial * 5 < time_consolidate_full

            # Numbers for root (apart from flag summaries) must not change
            assert_series_equal(
                tree3.drop(columns=FLAG_SUMMARY_NAMES).loc[root_id],
                tree0.drop(columns=FLAG_SUMMARY_NAMES).loc[root_id],
            )

            ## Partial rebuild and full rebuild should be the same
            # Assert tree2 == tree3 for relevant properties
            assert list(tree2.columns) == list(tree3.columns)
            # print(list(tree2.columns))
            assert_frame_equal(tree2, tree3, check_dtype=False)
            assert_vectors_equal(tree2.index, vector_own_2, vector_own_3)
            # TODO: Increase precision by changing nodes.vector_own_ back to Pickle type
            assert_vectors_equal(tree2.index, vector_2, vector_3, atol=1e-4)

            # TODO: Assert tree1 (with invalid values) == tree3 (full rebuild) for relevant properties apart from nan entries

        txn.rollback()


def _prepare_tree(tree, vector_columns, drop_columns):
    tree = tree.sort_index()

    def convert_column(c):
        try:
            return seq2array(tree[c])
        except:
            print("Could not convert column {}".format(c))
            raise

    vec_data = [convert_column(c) for c in vector_columns]
    tree = tree.drop(columns=vector_columns + drop_columns)

    return tree, vec_data


# def compare_tree(a, b, vector_columns, drop_columns):
#     a, vec_a = _prepare_tree(a, vector_columns, drop_columns)
#     b, vec_b = _prepare_tree(b, vector_columns, drop_columns)

#     assert_frame_equal(a, b)

#     assert_allclose(vector_own_2, vector_own_3, atol=1e-5)

#     assert_allclose(vector_2, vector_3, atol=1e-4)
