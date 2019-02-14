import itertools
import os
from getpass import getpass

import click
import flask_migrate
import h5py
import pandas as pd
from etaprogress.progress import ProgressBar
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.expression import bindparam
from timer_cm import Timer
from werkzeug.security import generate_password_hash

from morphocluster import models
from morphocluster.extensions import database
from morphocluster.tree import Tree


def _add_user(username, password):
    pwhash = generate_password_hash(
        password, method='pbkdf2:sha256:10000', salt_length=12)

    with database.engine.connect() as conn:
        stmt = models.users.insert(
            {"username": username, "pwhash": pwhash})
        conn.execute(stmt)


def init_app(app):
    @app.cli.command()
    def reset_db():
        """
        Drop all tables and recreate.
        """
        print("Resetting the database.")
        print("WARNING: This is a destructive operation and all data will be lost.")

        if input("Continue? (y/n) ") != "y":
            print("Canceled.")
            return

        with database.engine.begin() as txn:
            database.metadata.drop_all(txn)
            database.metadata.create_all(txn)

            flask_migrate.stamp()

    @app.cli.command()
    def clear_cache():
        """
        Clear cached values.
        """
        with database.engine.begin() as txn:
            # Cached values are prefixed with an underscore
            cached_columns = list(
                c for c in models.nodes.columns.keys() if c.startswith("_"))
            values = {c: None for c in cached_columns}
            values["cache_valid"] = False
            stmt = models.nodes.update().values(values)
            txn.execute(stmt)

        print("Cache was cleared.")

    @app.cli.command()
    def clear_projects():
        """
        Delete all project-related data.
        """
        print("Clearing projects.")
        print("WARNING: This is a destructive operation and all data will be lost.")

        if input("Continue? (y/n) ") != "y":
            print("Canceled.")
            return

        # TODO: Cascade drop https://stackoverflow.com/a/38679457/1116842

        print("Clearing project data...")
        with database.engine.begin() as txn:
            affected_tables = [models.nodes,
                               models.projects, models.nodes_objects]
            database.metadata.drop_all(txn, tables=affected_tables)
            database.metadata.create_all(txn, tables=affected_tables)

    @app.cli.command()
    @click.argument('collection_fn')
    def load_object_locations(collection_fn):
        """
        Load a collection of objects.
        """
        # Load collections
        with database.engine.begin() as txn:
            print("Loading {}...".format(collection_fn))
            data = pd.read_csv(collection_fn,
                               header=None,
                               names=["object_id", "path", "label"],
                               usecols=["object_id", "path"])

            data_iter = data.itertuples()
            bar = ProgressBar(len(data), max_width=40)
            while True:
                chunk = tuple(itertools.islice(data_iter, 5000))
                if not chunk:
                    break
                txn.execute(models.objects.insert(),
                            [row._asdict() for row in chunk])

                bar.numerator += len(chunk)
                print(bar, end="\r")
            print()
            print("Done.")

    @app.cli.command()
    @click.argument('features_fns', nargs=-1)
    def load_features(features_fns):
        """
        Load object features from an HDF5 file.
        """
        for features_fn in features_fns:
            print("Loading {}...".format(features_fn))
            with h5py.File(features_fn, "r", libver="latest") as f_features, database.engine.begin() as conn:
                object_ids = f_features["objids"]
                vectors = f_features["features"]

                stmt = (models.objects.update()
                        .where(models.objects.c.object_id == bindparam('_object_id'))
                        .values({
                            'vector': bindparam('vector')
                        }))

                bar = ProgressBar(len(object_ids), max_width=40)
                obj_iter = iter(zip(object_ids, vectors))
                while True:
                    chunk = tuple(itertools.islice(obj_iter, 1000))
                    if not chunk:
                        break
                    conn.execute(stmt, [{"_object_id": str(object_id), "vector": vector} for (
                        object_id, vector) in chunk])

                    bar.numerator += len(chunk)
                    print(bar, end="\r")
                print()
                print("Done.")

    @app.cli.command()
    @click.argument('tree_fn')
    @click.argument('project_name', default=None)
    @click.option('--consolidate/--no-consolidate', default=True)
    def load_project(tree_fn, project_name, consolidate):
        """
        Load a project from a saved tree.
        """

        with database.engine.connect() as conn:
            tree = Tree(conn)

            if project_name is None:
                project_name = os.path.basename(os.path.splitext(tree_fn)[0])

            with conn.begin():
                print("Loading {}...".format(
                    tree_fn))
                project_id = tree.load_project(
                    project_name, tree_fn)
                root_id = tree.get_root_id(project_id)

                if consolidate:
                    print("Consolidating ...")
                    tree.consolidate_node(root_id)

            print("Root ID: {}".format(root_id))
            print("Project ID: {}".format(project_id))

    @app.cli.command()
    @click.argument('root_id', type=int)
    @click.argument('tree_fn')
    def export_tree(root_id, tree_fn):
        """
        Export the whole tree with its objects.
        """
        with database.engine.connect() as conn:
            tree = Tree(conn)

            tree.export_tree(root_id, tree_fn)

    @app.cli.command()
    @click.argument('root_id', type=int, required=False)
    @click.option('--log/--no-log', "log", default=False)
    def progress(root_id, log):
        """
        Report progress on a tree
        """
        with database.engine.connect() as conn:
            tree = Tree(conn)

            if root_id is None:
                root_ids = [p["node_id"] for p in tree.get_projects()]
            else:
                root_ids = [root_id]

            with Timer("Progress") as timer:
                for rid in root_ids:
                    print("Root {}:".format(rid))
                    with timer.child(str(rid)):
                        prog = tree.calculate_progress(rid)

                    for k in sorted(prog.keys()):
                        print("{}: {}".format(k, prog[k]))

    @app.cli.command()
    @click.argument('root_id', type=int)
    def connect_supertree(root_id):
        with database.engine.connect() as conn:
            tree = Tree(conn)
            tree.connect_supertree(root_id)

    @app.cli.command()
    @click.argument('root_id', type=int, required=False)
    def consolidate(root_id):
        with database.engine.connect() as conn, Timer("Consolidate") as timer:
            tree = Tree(conn)

            if root_id is None:
                root_ids = [p["node_id"] for p in tree.get_projects()]
            else:
                root_ids = [root_id]

            for rid in root_ids:
                with timer.child(str(rid)):
                    tree.consolidate_node(rid)

    @app.cli.command()
    @click.argument('username')
    def add_user(username):
        print("Adding user {}:".format(username))
        password = getpass("Password: ")
        password_repeat = getpass("Retype Password: ")

        if not len(password):
            print("Password must not be empty!")
            return

        if password != password_repeat:
            print("Passwords do not match!")
            return

        try:
            _add_user(username, password)
        except IntegrityError as e:
            print(e)

    @app.cli.command()
    @click.argument('username')
    def change_user(username):
        print("Changing user {}:".format(username))
        password = getpass("Password: ")
        password_repeat = getpass("Retype Password: ")

        if password != password_repeat:
            print("Passwords do not match!")
            return

        pwhash = generate_password_hash(
            password, method='pbkdf2:sha256:10000', salt_length=12)

        try:
            with database.engine.connect() as conn:
                stmt = models.users.insert(
                    {"username": username, "pwhash": pwhash})
                conn.execute(stmt)
        except IntegrityError as e:
            print(e)

    @app.cli.command()
    @click.argument('root_id')
    @click.argument('classification_fn')
    def export_classifications(root_id, classification_fn):
        with database.engine.connect() as conn:
            tree = Tree(conn)
            tree.export_classifications(root_id, classification_fn)

    @app.cli.command()
    @click.argument('node_id')
    @click.argument('filename')
    def export_direct_objects(node_id, filename):
        with database.engine.connect() as conn, open(filename, "w") as f:
            tree = Tree(conn)

            f.writelines("{}\n".format(o["object_id"])
                         for o in tree.get_objects(node_id))

    @app.cli.command()
    @click.argument('filename')
    def export_log(filename):
        with database.engine.connect() as conn:
            log = pd.read_sql_query(
                models.log.select(), conn, index_col="log_id")
            log.to_csv(filename)
