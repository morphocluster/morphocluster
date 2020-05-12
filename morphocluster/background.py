import flask_rq2

from morphocluster.extensions import database, rq
from morphocluster.tree import Tree
import os
import datetime as dt
from morphocluster.processing.recluster import Recluster
from flask import current_app as app


def validate_background_job(fun):
    return isinstance(getattr(fun, "helper", None), flask_rq2.functions.JobFunctions)


@rq.job
def add(x, y):
    return x + y


@rq.job
def export_project(project_id):
    config = app.config

    # Dump the database tree
    with database.engine.connect() as conn:
        db_tree = Tree(conn)
        root_id = db_tree.get_root_id(project_id)
        project = db_tree.get_project(project_id)
        tree = db_tree.dump_tree(root_id)

    tree_fn = os.path.join(
        config["PROJECT_EXPORT_DIR"],
        "{:%Y-%m-%d-%H-%M-%S}--{}--{}.zip".format(
            dt.datetime.now(), project["project_id"], project["name"]
        ),
    )

    tree.save(tree_fn)

    return tree_fn


@rq.job(timeout=43200)
def recluster_project(project_id, min_cluster_size):
    """
    Timeout: 12h
    """

    config = morphocluster.app.app.config

    # Dump the database tree
    print("Dumping database tree...")
    with database.engine.connect() as conn:
        db_tree = Tree(conn)
        root_id = db_tree.get_root_id(project_id)
        project = db_tree.get_project(project_id)
        tree = db_tree.dump_tree(root_id)

    # Recluster unapproved objects
    print("Reclustering...")
    recluster = Recluster()
    recluster.load_tree(tree)

    for features_fn in config["RECLUSTER_FEATURES"]:
        recluster.load_features(features_fn)

    # Cluster 1M objects maximum
    # sample_size = int(1e6)
    sample_size = 1000

    recluster.cluster(
        ignore_approved=True,
        sample_size=sample_size,
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_method="leaf",
    )

    tree = recluster.merge_trees()

    # Load new tree into the database
    print("Loading tree into database...")
    project_name = "{}-{}".format(project["name"], min_cluster_size)

    with database.engine.connect() as conn:
        db_tree = Tree(conn)

        with conn.begin():
            project_id = db_tree.load_project(project_name, tree)
            root_id = db_tree.get_root_id(project_id)

            print("Consolidating ...")
            db_tree.consolidate_node(root_id)

        print("Root ID: {}".format(root_id))
        print("Project ID: {}".format(project_id))

    print("Done.")
