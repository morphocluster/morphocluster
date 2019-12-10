import os.path

import h5py
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
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


@pytest.fixture(name="project")
def _project(dataset: Dataset, datadir):
    print("Creating project...")
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


# relocate_nodes has to keep all these values valid
# n_nodes_ can not be calculated reliably.
VALID_AFTER_RELOCATE_NODES = (
    "n_objects_own_",
    "n_objects_",
    "n_children_",
    "vector_sum_own_",
    "vector_sum_",
)

# These vales needs to be invalidated for affected precursers
AP_INVALID_AFTER_RELOCATE_NODES = (
    "n_approved_objects_",
    "n_filled_objects_",
    "n_preferred_objects_",
    "n_approved_nodes_",
    "n_filled_nodes_",
    "n_preferred_nodes_",
    "vector_mean_",
)


def assert_all_valid_after_consolidate(tree):
    assert_all_cache_valid(tree)
    assert_all_valid(
        tree,
        (
            "n_children_",
            "n_objects_own_",
            "n_objects_",
            "n_approved_objects_",
            "n_approved_nodes_",
            "n_filled_objects_",
            "n_filled_nodes_",
            "n_preferred_objects_",
            "n_preferred_nodes_",
            "vector_sum_",
            "vector_sum_own_",
        ),
    )
    assert_all_valid_iff(
        tree, tree["n_objects_"] > 0, ("type_objects_", "vector_mean_")
    )
    assert_all_valid_iff(
        tree, tree["n_objects_own_"] > 0, ("type_objects_own_", "vector_mean_own_")
    )


# Assert that consolidate_node works as expected
def test_consolidate_node_raw(project, orig_tree, datadir):
    features_fn = str(datadir / "features.h5")
    with project, h5py.File(features_fn, "r") as f_features:
        object_ids = f_features["object_id"]
        vectors = f_features["features"]

        root_id = project.get_root_id()
        result = project.consolidate_node(
            root_id, depth=-1, return_="raw", exact_vector="exact"
        )

        assert_all_valid_after_consolidate(result)

        for node_id, values in result.iterrows():
            ## Assert that n_children_ is correct
            n_children_orig = (orig_tree.nodes["parent_id"] == node_id).sum()
            assert n_children_orig == values["n_children_"]

            ## Assert that n_objects_own_ is correct
            n_objects_own_orig = (orig_tree.objects["node_id"] == node_id).sum()
            assert n_objects_own_orig == values["n_objects_own_"]

            ## Assert that vector_sum_own_ is correct
            object_selector = orig_tree.objects["node_id"] == node_id
            node_object_ids = orig_tree.objects.loc[object_selector, "object_id"]
            h5_selector = pd.Series(object_ids).isin(node_object_ids)

            if h5_selector.any():
                vector_sum_own_h5 = vectors[h5_selector].sum(axis=0)
                assert_allclose(
                    vector_sum_own_h5,
                    values["vector_sum_own_"],
                    atol=1e-6,
                    err_msg="Unexpected vector_sum_own_ for node_id={}".format(node_id),
                )
            else:
                assert values["vector_sum_own_"] is 0

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


def assert_vectors_equal(node_ids, vec_a, vec_b, obj="Vectors", atol=1e-5):
    __tracebackhide__ = True  # pylint: disable=unused-variable

    dist = np.linalg.norm(vec_a - vec_b, axis=1)
    assert len(dist) == len(node_ids)
    max_dist = np.max(dist)
    dist_ok = dist <= atol
    p_match = np.mean(dist_ok)
    n_match = np.sum(dist_ok)
    n = len(dist_ok)
    n_miss = n - n_match
    assert dist_ok.all(), "{} match {:.3%} ({:d}+{:d}={:d}). Distances: {}".format(
        obj,
        p_match,
        n_match,
        n_miss,
        n,
        [{nid: d} for nid, d in zip(node_ids[~dist_ok], dist[~dist_ok])],
    )
    print("Max dist of {}: {:.2g}".format(obj, max_dist))


def test_commonprefix():
    assert commonprefix([[1, 2, 3], [1, 2, 4, 5], [1, 2, 6, 7, 8]]) == [1, 2]
    assert commonprefix([[1, 2, 3], [1, 4, 5], [1, 6, 7, 8]]) == [1]
    assert commonprefix([[1], [1, 2, 3], [1, 4, 5], [1, 6, 7, 8]]) == []


def assert_all_valid(df, columns, extra_cb=None):
    __tracebackhide__ = True  # pylint: disable=unused-variable

    for col in columns:
        valid = ~pd.isna(df[col])
        fails_for = df[~valid].index
        extra_info = extra_cb(fails_for) if extra_cb is not None else None
        assert (
            valid.all()
        ), "{col} not valid for {n_invalid:d} indexes {indexes} ({extra_info})".format(
            col=col,
            n_invalid=(~valid).sum(),
            indexes=fails_for.values,
            extra_info=extra_info,
        )


def assert_all_invalid(df, columns, extra_cb=None):
    __tracebackhide__ = True  # pylint: disable=unused-variable

    for col in columns:
        invalid = pd.isna(df.loc[:, col])
        fails_for = df[~invalid].index
        extra_info = extra_cb(fails_for) if extra_cb is not None else None
        assert (
            invalid.all()
        ), "{col} not invalid for {n_valid:d} indexes {indexes} ({extra_info})".format(
            col=col,
            n_valid=(~invalid).sum(),
            indexes=fails_for.values,
            extra_info=extra_info,
        )


def assert_all_valid_iff(df, condition, columns, extra_cb=None):
    __tracebackhide__ = True

    for col in columns:
        valid = ~pd.isna(df[col])
        equal = condition == valid
        fails_for = df[~equal].index
        assert (
            equal.all()
        ), "{col} validity does not match condition for {n_neq:d} indexes {indexes} ({extra_info}):\ncondition: {condition}\nvalid: {valid}".format(
            indexes=fails_for.values,
            extra_info=extra_cb(fails_for) if extra_cb is not None else None,
            condition=condition[~equal].values,
            valid=valid[~equal].values,
            col=col,
            n_neq=(~equal).sum(),
        )


def test_relocate_objects(project: Project, orig_tree: pd.DataFrame):
    # Assert relocate_objects recalculates the vectors
    with project:
        root_id = project.get_root_id()
        n_objects = project.get_n_objects()
        n_nodes = project.get_n_nodes()

        # Ignore these columns in comparisons because they are indeterministic or insignificant
        IGNORE_COLUMNS = ["type_objects_own_", "type_objects_", "cache_valid"]

        # Reset all cached values for this project
        project.reset_cached_values()

        # Make sure tree is up to date and record tree before relocate
        tree0 = project.consolidate_node(
            root_id, depth=-1, exact_vector="exact", return_="raw"
        )

        # Assert that all calculated values are valid
        assert_all_valid_after_consolidate(tree0)

        tree0, _ = _prepare_tree(
            tree0,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
        )

        assert tree0.shape[0] == n_nodes
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

        # relocate_objects has to keep all these values valid
        assert_all_valid(
            tree1, ("n_objects_own_", "n_objects_", "vector_sum_own_", "vector_sum_")
        )

        tree1, (
            vector_sum_own_1,
            vector_sum_1,
            vector_mean_own_1,
            vector_mean_1,
        ) = _prepare_tree(
            tree1,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
        )

        # relocate_objects has to reset all flag summaries for all predecessors
        for flag_summary_name in FLAG_SUMMARY_NAMES:
            assert pd.isna(
                tree1.loc[affected_precursers, flag_summary_name]
            ).all(), "{} has to be reset for all predecessors".format(flag_summary_name)

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

        assert_all_valid_after_consolidate(tree2)

        tree2, (
            vector_sum_own_2,
            vector_sum_2,
            vector_mean_own_2,
            vector_mean_2,
        ) = _prepare_tree(
            tree2,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
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

        tree3, (
            vector_sum_own_3,
            vector_sum_3,
            vector_mean_own_3,
            vector_mean_3,
        ) = _prepare_tree(
            tree3,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
        )

        # TODO: Assert that partial consolidate is faster
        # assert time_consolidate_partial * 5 < time_consolidate_full

        # Numbers for root (apart from flag summaries) must not change
        assert_series_equal(
            tree3.drop(columns=FLAG_SUMMARY_NAMES).loc[root_id],
            tree0.drop(columns=FLAG_SUMMARY_NAMES).loc[root_id],
        )

        ## Assert tree1 (with invalid values) == tree3 (full rebuild) for relevant properties apart from nan entries
        invalidated_columns = [
            "n_approved_objects_",
            "n_filled_objects_",
            "n_preferred_objects_",
        ]
        assert_frame_equal(
            tree1.drop(columns=invalidated_columns),
            tree3.drop(columns=invalidated_columns),
            check_dtype=False,
        )
        # Vectors after relocate should be the same as full rebuild
        assert_vectors_equal(
            tree1.index,
            vector_sum_own_1,
            vector_sum_own_3,
            atol=1e-4,
            obj="vector_sum_own_1:vector_sum_own_3",
        )
        assert_vectors_equal(
            tree1.index,
            vector_sum_1,
            vector_sum_3,
            atol=1e-4,
            obj="vector_sum_1:vector_sum_3",
        )

        ## Partial rebuild and full rebuild should be the same
        # Assert tree2 == tree3 for relevant properties
        assert list(tree2.columns) == list(tree3.columns)
        # print(list(tree2.columns))
        assert_frame_equal(tree2, tree3, check_dtype=False)
        assert_vectors_equal(
            tree2.index,
            vector_sum_own_2,
            vector_sum_own_3,
            obj="vector_sum_own_2:vector_sum_own_3",
        )
        # TODO: Increase precision by changing nodes.vector_sum_own_ back to Pickle type
        assert_vectors_equal(
            tree2.index,
            vector_sum_2,
            vector_sum_3,
            atol=1e-4,
            obj="vector_sum_2:vector_sum_3",
        )


def _prepare_tree(tree, vector_sum_columns, drop_columns):
    tree = tree.sort_index()

    def convert_column(c):
        try:
            return seq2array(tree[c])
        except:
            print("Could not convert column {}".format(c))
            raise

    vec_data = [convert_column(c) for c in vector_sum_columns]
    tree = tree.drop(columns=vector_sum_columns + drop_columns)

    return tree, vec_data


# def compare_tree(a, b, vector_sum_columns, drop_columns):
#     a, vec_a = _prepare_tree(a, vector_sum_columns, drop_columns)
#     b, vec_b = _prepare_tree(b, vector_sum_columns, drop_columns)

#     assert_frame_equal(a, b)

#     assert_allclose(vector_sum_own_2, vector_sum_own_3, atol=1e-5)

#     assert_allclose(vector_sum_2, vector_sum_3, atol=1e-4)


def test_update_node():
    ...


def test_merge_node(project: Project):
    # Assert relocate_objects recalculates the vectors
    with project:
        root_id = project.get_root_id()
        n_objects = project.get_n_objects()
        n_nodes = project.get_n_nodes()

        # Ignore these columns in comparisons because they are indeterministic or insignificant
        IGNORE_COLUMNS = ["type_objects_own_", "type_objects_", "cache_valid"]

        # Reset all cached values for this project
        project.reset_cached_values()

        # Make sure tree is up to date and record tree before relocate
        tree0 = project.consolidate_node(
            root_id, depth=-1, exact_vector="exact", return_="raw"
        )

        # Assert that all calculated values are valid
        assert_all_valid_after_consolidate(tree0)

        tree0, _ = _prepare_tree(
            tree0,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
        )

        assert tree0.shape[0] == n_nodes
        assert tree0.loc[root_id, "n_objects_"] == n_objects

        # Sample dest_node_id
        mask = tree0["n_nodes_"] > 10
        dest_node_id = int(tree0[mask].sample().index[0])

        # Sample node_id from subtree below dest_node_id
        subtree = project.get_subtree(dest_node_id)
        mask = subtree.index != dest_node_id
        node_id = int(subtree[mask].sample().index[0])

        old_path = project.get_path(node_id)
        dest_node_path = project.get_path(dest_node_id)

        assert node_id not in dest_node_path

        n_children = (subtree["parent_id"] == node_id).sum()

        affected_precursers = set(old_path)
        affected_precursers.update(dest_node_path)

        prefix = commonprefix([old_path, dest_node_path])
        len_prefix = len(prefix)
        affected_subtree_nodes = set(old_path[len_prefix:])
        affected_subtree_nodes.update(dest_node_path[len_prefix:])

        affected_precursers = sorted(affected_precursers)

        print("{} affected precursors.".format(len(affected_precursers)))
        print("{} affected subtree nodes.".format(len(affected_subtree_nodes)))

        ## Merge node
        print("Merging {} into {}...".format(node_id, dest_node_id))
        n_relocated_objects, n_relocated_children = project.merge_node_into(
            node_id, dest_node_id
        )

        print("Relocated {} objects.".format(n_relocated_objects))
        print("Relocated {} children.".format(n_relocated_children))
        assert n_relocated_children == n_children

        # The flag summaries change between before and after merge
        FLAG_SUMMARY_NAMES = [
            "n_approved_objects_",
            "n_filled_objects_",
            "n_preferred_objects_",
            "n_approved_nodes_",
            "n_filled_nodes_",
            "n_preferred_nodes_",
        ]

        # Record tree after merge
        tree1 = project.get_subtree(root_id)

        ## Assert that node_id is actually gone
        assert node_id not in tree1.index

        # merge_node_into has to keep all these values valid
        # n_nodes_ can not be calculated reliably.
        assert_all_valid(
            tree1,
            (
                "n_objects_own_",
                "n_objects_",
                "n_children_",
                "vector_sum_own_",
                "vector_sum_",
            ),
        )

        tree1, (
            vector_sum_own_1,
            vector_sum_1,
            vector_mean_own_1,
            vector_mean_1,
        ) = _prepare_tree(
            tree1,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
        )

        # merge_node_into does not necessarily reset all affected_precursers,
        # e.g. when the merged node has no objects
        if n_relocated_objects:
            assert_all_invalid(
                tree1.loc[affected_precursers],
                ("n_approved_objects_", "n_filled_objects_", "n_preferred_objects_"),
            )

        assert_all_invalid(
            tree1.loc[affected_precursers],
            ("n_approved_nodes_", "n_filled_nodes_", "n_preferred_nodes_"),
        )

        # assert_all_invalid(tree1.loc[affected_precursers], FLAG_SUMMARY_NAMES)

        ## Assert that certain properties for root don't change
        # These properties are affected by merge_node_into, ignore.
        variants_for_root = FLAG_SUMMARY_NAMES + ["n_nodes_", "n_children_"]
        assert_series_equal(
            tree1.drop(columns=variants_for_root).loc[root_id],
            tree0.drop(columns=variants_for_root).loc[root_id],
            obj="Series({})".format(
                ",".join(tree1.drop(columns=variants_for_root).columns)
            ),
        )

        ## consolidate to fill in the holes left by merge_node_into
        with Timer("consolidate_partial") as t:
            tree2 = project.consolidate_node(
                root_id, depth=-1, exact_vector="exact", return_="raw"
            )
        time_consolidate_partial = t.elapsed

        assert_all_valid_after_consolidate(tree2)

        tree2, (
            vector_sum_own_2,
            vector_sum_2,
            vector_mean_own_2,
            vector_mean_2,
        ) = _prepare_tree(
            tree2,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
        )

        ## Assert that certain properties for root don't change
        assert_series_equal(
            tree2.drop(columns=variants_for_root).loc[root_id],
            tree0.drop(columns=variants_for_root).loc[root_id],
            obj="Series({})".format(
                ",".join(tree2.drop(columns=variants_for_root).columns)
            ),
        )

        ## Reset and consolidate again to compare to fresh values
        project.reset_cached_values()
        with Timer("consolidate_full") as t:
            tree3 = project.consolidate_node(
                root_id, depth=-1, exact_vector="exact", return_="raw"
            )
        time_consolidate_full = t.elapsed

        tree3, (
            vector_sum_own_3,
            vector_sum_3,
            vector_mean_own_3,
            vector_mean_3,
        ) = _prepare_tree(
            tree3,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
        )

        # TODO: Assert that partial consolidate is faster
        # assert time_consolidate_partial * 5 < time_consolidate_full

        ## Assert that certain properties for root don't change
        assert_series_equal(
            tree3.drop(columns=variants_for_root).loc[root_id],
            tree0.drop(columns=variants_for_root).loc[root_id],
            obj="Series({})".format(
                ",".join(tree3.drop(columns=variants_for_root).columns)
            ),
        )

        ## Assert tree1 (with invalid values) == tree3 (full rebuild) for relevant properties apart from nan entries
        invalidated_columns = [
            "n_approved_objects_",
            "n_filled_objects_",
            "n_preferred_objects_",
            "n_approved_nodes_",
            "n_filled_nodes_",
            "n_preferred_nodes_",
        ]
        # n_nodes_ can not be calculated reliably
        assert_frame_equal2(
            tree1.drop(columns=invalidated_columns + ["n_nodes_"]),
            tree3.drop(columns=invalidated_columns + ["n_nodes_"]),
            lambda fails_for: {
                node_id: "node_id",
                dest_node_id: "dest_node_id",
                root_id: "root_id",
            }.get(fails_for[0], "no match"),
        )
        # Vectors after relocate should be the same as full rebuild
        assert_vectors_equal(
            tree1.index,
            vector_sum_own_1,
            vector_sum_own_3,
            atol=1e-4,
            obj="vector_sum_own_1:vector_sum_own_3",
        )
        assert_vectors_equal(
            tree1.index,
            vector_sum_1,
            vector_sum_3,
            atol=1e-4,
            obj="vector_sum_1:vector_sum_3",
        )

        ## Partial rebuild and full rebuild should be the same
        # Assert tree2 == tree3 for relevant properties
        assert list(tree2.columns) == list(tree3.columns)
        # print(list(tree2.columns))
        assert_frame_equal(tree2, tree3, check_dtype=False)
        assert_vectors_equal(
            tree2.index,
            vector_sum_own_2,
            vector_sum_own_3,
            obj="vector_sum_own_2:vector_sum_own_3",
        )
        # TODO: Increase precision by changing nodes.vector_sum_own_ back to Pickle type
        assert_vectors_equal(
            tree2.index,
            vector_sum_2,
            vector_sum_3,
            atol=1e-4,
            obj="vector_sum_2:vector_sum_3",
        )


RELOCATE_NODES_INVARIANT_ROOT_BEFORE_AFTER = [
    "project_id",
    "parent_id",
    "name",
    "starred",
    "approved",
    "filled",
    "preferred",
    "n_objects_own_",
    "n_objects_",
]


def test_relocate_nodes(project: Project):
    with project:
        root_id = project.get_root_id()
        n_objects = project.get_n_objects()
        n_nodes = project.get_n_nodes()

        # Ignore these columns in comparisons because they are indeterministic or insignificant
        IGNORE_COLUMNS = ["type_objects_own_", "type_objects_", "cache_valid"]

        # Reset all cached values for this project
        project.reset_cached_values()

        # Make sure tree is up to date and record tree before relocate
        tree0 = project.consolidate_node(
            root_id, depth=-1, exact_vector="exact", return_="raw"
        )

        # Assert that all calculated values are valid
        assert_all_valid_after_consolidate(tree0)

        tree0, _ = _prepare_tree(
            tree0,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
        )

        assert tree0.shape[0] == n_nodes
        assert tree0.loc[root_id, "n_objects_"] == n_objects

        # Sample new_parent_id
        mask = tree0["n_nodes_"] > 20
        new_parent_id = int(tree0[mask].sample().index[0])

        # Sample node_ids from subtree below new_parent_id
        subtree = project.get_subtree(new_parent_id)
        mask = subtree.index != new_parent_id
        node_ids = [int(i) for i in subtree[mask].sample(10).index]

        old_paths = [project.get_path(int(nid)) for nid in node_ids]
        new_parent_path = project.get_path(new_parent_id)

        affected_precursers = set(sum(old_paths, []))
        affected_precursers.update(new_parent_path)

        assert all(i not in new_parent_path for i in node_ids)

        prefix = commonprefix(old_paths + [new_parent_path])
        len_prefix = len(prefix)
        affected_subtree_nodes = set(sum((p[len_prefix:] for p in old_paths), []))
        affected_subtree_nodes.update(new_parent_path[len_prefix:])

        affected_precursers = sorted(affected_precursers)

        print("{} affected precursors.".format(len(affected_precursers)))
        print("{} affected subtree nodes.".format(len(affected_subtree_nodes)))

        ## Relocate nodes
        n_relocated_nodes = project.relocate_nodes(node_ids, new_parent_id)

        assert n_relocated_nodes == len(node_ids)

        # The flag summaries change between before and after relocate
        FLAG_SUMMARY_NAMES = [
            "n_approved_objects_",
            "n_filled_objects_",
            "n_preferred_objects_",
            "n_approved_nodes_",
            "n_filled_nodes_",
            "n_preferred_nodes_",
        ]

        # Record tree after relocate_nodes
        tree1 = project.get_subtree(root_id)

        ## Assert that a new parent_id was actually set
        assert (tree1.loc[node_ids, "parent_id"] == new_parent_id).all()

        ## Assert that these values are valid
        assert_all_valid(tree1, VALID_AFTER_RELOCATE_NODES)

        ## Assert that these values are reset for affected precursors
        assert_all_invalid(
            tree1.loc[affected_precursers], AP_INVALID_AFTER_RELOCATE_NODES
        )

        tree1, (
            vector_sum_own_1,
            vector_sum_1,
            vector_mean_own_1,
            vector_mean_1,
        ) = _prepare_tree(
            tree1,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
        )

        ## Assert that certain properties for root don't change
        assert_series_equal(
            tree1.loc[root_id, RELOCATE_NODES_INVARIANT_ROOT_BEFORE_AFTER],
            tree0.loc[root_id, RELOCATE_NODES_INVARIANT_ROOT_BEFORE_AFTER],
            obj="Series({})".format(
                ",".join(tree1[RELOCATE_NODES_INVARIANT_ROOT_BEFORE_AFTER].columns)
            ),
        )

        ## consolidate to fill in the holes left by merge_node_into
        with Timer("consolidate_partial") as t:
            tree2 = project.consolidate_node(
                root_id, depth=-1, exact_vector="exact", return_="raw"
            )
        time_consolidate_partial = t.elapsed

        assert_all_valid_after_consolidate(tree2)

        tree2, (
            vector_sum_own_2,
            vector_sum_2,
            vector_mean_own_2,
            vector_mean_2,
        ) = _prepare_tree(
            tree2,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
        )

        ## Assert that certain properties for root don't change
        assert_series_equal(
            tree2.loc[root_id, RELOCATE_NODES_INVARIANT_ROOT_BEFORE_AFTER],
            tree0.loc[root_id, RELOCATE_NODES_INVARIANT_ROOT_BEFORE_AFTER],
            obj="Series({})".format(
                ",".join(tree1[RELOCATE_NODES_INVARIANT_ROOT_BEFORE_AFTER].columns)
            ),
        )

        ## Reset and consolidate again to compare to fresh values
        project.reset_cached_values()
        with Timer("consolidate_full") as t:
            tree3 = project.consolidate_node(
                root_id, depth=-1, exact_vector="exact", return_="raw"
            )
        time_consolidate_full = t.elapsed

        tree3, (
            vector_sum_own_3,
            vector_sum_3,
            vector_mean_own_3,
            vector_mean_3,
        ) = _prepare_tree(
            tree3,
            ["vector_sum_own_", "vector_sum_", "vector_mean_own_", "vector_mean_"],
            IGNORE_COLUMNS,
        )

        # TODO: Assert that partial consolidate is faster
        # assert time_consolidate_partial * 5 < time_consolidate_full

        ## Assert that certain properties for root don't change
        assert_series_equal(
            tree3.loc[root_id, RELOCATE_NODES_INVARIANT_ROOT_BEFORE_AFTER],
            tree0.loc[root_id, RELOCATE_NODES_INVARIANT_ROOT_BEFORE_AFTER],
            obj="Series({})".format(
                ",".join(tree1[RELOCATE_NODES_INVARIANT_ROOT_BEFORE_AFTER].columns)
            ),
        )

        ## Assert tree1 (with invalid values) == tree3 (full rebuild) for relevant properties apart from nan entries
        invalidated_columns = [
            "n_approved_objects_",
            "n_filled_objects_",
            "n_preferred_objects_",
            "n_approved_nodes_",
            "n_filled_nodes_",
            "n_preferred_nodes_",
        ]

        # n_nodes_ can not be calculated reliably
        assert_frame_equal2(
            tree1.drop(columns=invalidated_columns + ["n_nodes_"]),
            tree3.drop(columns=invalidated_columns + ["n_nodes_"]),
            lambda fails_for: fails_for.isin(node_ids),
        )

        # Vectors after relocate_nodes should be the same as full rebuild
        assert_vectors_equal(
            tree1.index,
            vector_sum_own_1,
            vector_sum_own_3,
            atol=1e-4,
            obj="vector_sum_own_1:vector_sum_own_3",
        )
        assert_vectors_equal(
            tree1.index,
            vector_sum_1,
            vector_sum_3,
            atol=1e-4,
            obj="vector_sum_1:vector_sum_3",
        )

        ## Partial rebuild and full rebuild should be the same
        # Assert tree2 == tree3 for relevant properties
        assert list(tree2.columns) == list(tree3.columns)
        # print(list(tree2.columns))
        # assert_frame_equal(tree2, tree3, check_dtype=False)
        assert_frame_equal2(tree2, tree3, lambda fails_for: fails_for.isin(node_ids))
        assert_vectors_equal(
            tree2.index,
            vector_sum_own_2,
            vector_sum_own_3,
            obj="vector_sum_own_2:vector_sum_own_3",
        )
        # TODO: Increase precision by changing nodes.vector_sum_own_ back to Pickle type
        assert_vectors_equal(
            tree2.index,
            vector_sum_2,
            vector_sum_3,
            atol=1e-4,
            obj="vector_sum_2:vector_sum_3",
        )


def assert_frame_equal2(a, b, extra_cb=None):
    __tracebackhide__ = True  # pylint: disable=unused-variable

    for i, col in enumerate(a.columns):
        assert col in b.columns
        equal = a.iloc[:, i] == b.iloc[:, i]
        equal[pd.isnull(a.iloc[:, i]) & pd.isnull(b.iloc[:, i])] = True  # NaN == NaN
        fails_for = a[~equal].index
        assert (
            equal.all()
        ), "{col} not equal for {n_neq:d} indexes {indexes} ({extra_info}):\nleft: {left}\nright: {right}".format(
            indexes=fails_for.values,
            extra_info=extra_cb(fails_for) if extra_cb is not None else None,
            left=a[~equal].iloc[:, i].values,
            right=b[~equal].iloc[:, i].values,
            col=col,
            n_neq=(~equal).sum(),
        )


def assert_all_cache_valid(tree, extra_cb=None):
    __tracebackhide__ = True  # pylint: disable=unused-variable

    cache_valid = tree["cache_valid"]
    fails_for = tree[~cache_valid].index
    extra_info = extra_cb(fails_for) if extra_cb is not None else None
    assert cache_valid.all(), "Cache not valid for indexes {} ({})".format(
        fails_for.values, extra_info
    )
