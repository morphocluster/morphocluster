import itertools
import os
import time
import zipfile
from typing import Dict, List, Optional
from xmlrpc.client import Boolean

import click
import flask_migrate
import h5py
import numpy as np
import pandas as pd
import sklearn.decomposition
import sqlalchemy.engine
import tqdm
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.expression import bindparam, select
from timer_cm import Timer
from werkzeug.security import generate_password_hash

from morphocluster import models, processing
from morphocluster.extensions import database
from morphocluster.tree import Tree


def _add_user(username, password):
    pwhash = generate_password_hash(
        password, method="pbkdf2:sha256:10000", salt_length=12
    )

    with database.engine.connect() as conn:
        stmt = models.users.insert().values(username=username, pwhash=pwhash)
        conn.execute(stmt)


def init_app(app):
    # pylint: disable=unused-variable

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
                c for c in models.nodes.columns.keys() if c.startswith("_")
            )
            values: Dict[str, Optional[Boolean]] = {c: None for c in cached_columns}
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
            affected_tables = [models.nodes, models.projects, models.nodes_objects]
            database.metadata.drop_all(txn, tables=affected_tables)
            database.metadata.create_all(txn, tables=affected_tables)

    def _load_new_objects(
        index: pd.DataFrame, batch_size: int, conn, zf: zipfile.ZipFile, dst_root: str
    ):
        if not index.size:
            return

        print(f"Loading {len(index):,d} new objects...")
        index_iter = index.itertuples()
        progress = tqdm.tqdm(total=len(index), unit_scale=True)
        while True:
            chunk = tuple(
                row._asdict() for row in itertools.islice(index_iter, batch_size)
            )
            if not chunk:
                break

            chunk_len = len(chunk)

            conn.execute(
                models.objects.insert(),  # pylint: disable=no-value-for-parameter
                [dict(row) for row in chunk],
            )

            for row in chunk:
                zf.extract(row["path"], dst_root)

            progress.update(chunk_len)
        progress.close()

    def _update_existing_objects(
        index: pd.DataFrame, batch_size: int, conn, zf: zipfile.ZipFile, dst_root: str
    ):
        if not index.size:
            return

        stmt = (
            models.objects.update()
            .where(models.objects.c.object_id == bindparam("_object_id"))
            .values({"path": bindparam("path")})
        )

        print(f"Updating {len(index):,d} existing objects...")
        index_iter = index.itertuples()
        progress = tqdm.tqdm(total=len(index), unit_scale=True)
        while True:
            chunk = tuple(
                row._asdict() for row in itertools.islice(index_iter, batch_size)
            )
            if not chunk:
                break

            chunk_len = len(chunk)

            # Update path
            conn.execute(
                stmt,
                [
                    {"_object_id": str(row["object_id"]), "path": row["path"]}
                    for row in chunk
                ],
            )

            for row in chunk:
                zf.extract(row["path"], dst_root)

                if row["path"] != row["path_old"]:
                    try:
                        os.remove(row["path_old"])
                    except FileNotFoundError:
                        print("Missing previous image:", row["path_old"])
                        pass

            progress.update(chunk_len)
        progress.close()

    @app.cli.command()
    @click.argument("archive_fn")
    @click.option("--add/--no-add", help="Add new objects", default=True)
    @click.option("--update/--no-update", help="Update existing objects", default=True)
    def load_objects(archive_fn: str, add: bool, update: bool):
        """Load an archive of objects into the database."""

        batch_size = 1000

        dst_root = app.config["DATASET_PATH"]

        print(f"Loading {archive_fn} into {dst_root}...")
        with database.engine.begin() as conn, zipfile.ZipFile(archive_fn) as zf:
            with zf.open("index.csv") as f:
                index: pd.DataFrame = pd.read_csv(f, usecols=["object_id", "path"])  # type: ignore

            if not index["object_id"].is_unique:
                value_counts = index["object_id"].value_counts()
                info = str(value_counts[value_counts > 1])
                raise ValueError(f"object_id contains duplicate values:\n{info}")

            # Divide index into new and existing objects
            print("Filtering existing entries...")
            stmt = select([models.objects.c.object_id, models.objects.c.path])
            existing = pd.read_sql(stmt, conn)

            mask_existing = index["object_id"].isin(existing["object_id"])
            index_new = index[~mask_existing]
            index_update = index.merge(
                existing, how="inner", on="object_id", suffixes=(None, "_old")
            )

            print(f"{len(existing):,d} objects already present in the database.")

            if add:
                _load_new_objects(index_new, batch_size, conn, zf, dst_root)

            if update:
                _update_existing_objects(index_update, batch_size, conn, zf, dst_root)

            print("Done.")

    @app.cli.command()
    @click.argument("features_fns", nargs=-1)
    @click.option("--truncate", type=int)
    @click.option("--pca", type=int)
    @click.option(
        "--clear/--no-clear", help="Clear previous features before importing."
    )
    def load_features(
        features_fns: List[str],
        truncate: Optional[int],
        pca: Optional[int],
        clear: bool,
    ):
        """
        Load object features from an HDF5 file.
        """
        object_ids_list = []
        vectors_list = []

        if not features_fns:
            return

        # Load all features
        for features_fn in features_fns:
            print("Loading {}...".format(features_fn))
            with h5py.File(features_fn, "r", libver="latest") as f_features:
                object_ids_list.append(f_features["object_id"].asstr()[:])  # type: ignore

                if truncate is None:
                    _features = f_features["features"][:]  # type: ignore
                else:
                    _features = f_features["features"][:, :truncate]  # type: ignore

                vectors_list.append(_features)

        object_ids: np.ndarray = np.concatenate(object_ids_list)  # type: ignore
        del object_ids_list
        vectors: np.ndarray = np.concatenate(vectors_list)  # type: ignore
        del vectors_list

        n_obj, n_dim = vectors.shape
        print(f"Loaded {n_dim}d features for {n_obj:,d} objects.")

        if pca is not None:
            print(f"Performing PCA ({pca}d)...")
            start = time.perf_counter()
            pca_transformer = sklearn.decomposition.PCA(pca)
            vectors = pca_transformer.fit_transform(vectors)
            time_fit = time.perf_counter() - start
            print("Dimensionality reduction took {:.0f}s".format(time_fit))
            print("Explained variance ratio:", pca_transformer.explained_variance_ratio_.sum())
            del pca_transformer

        if vectors.shape[1] > 100:
            raise ValueError(
                "The features can not have more than 100 dimensions. Try --truncate or --pca."
            )

        print("Moving feature vectors to the database...")
        with database.engine.begin() as conn:
            conn: sqlalchemy.engine.Connection

            if clear:
                print("Clearing previous features...")
                stmt = models.objects.update().values({"vector": None})
                conn.execute(stmt)

            stmt = (
                models.objects.update()
                .where(models.objects.c.object_id == bindparam("_object_id"))
                .values({"vector": bindparam("vector")})
            )

            # TODO: Use UPDATE ... RETURNING to get the number of affected rows

            progress = tqdm.tqdm(total=len(object_ids), unit="obj", unit_scale=True)
            obj_iter = iter(zip(object_ids, vectors))  # type: ignore
            while True:
                chunk = tuple(itertools.islice(obj_iter, 1000))
                if not chunk:
                    break
                conn.execute(
                    stmt,
                    [
                        {"_object_id": str(object_id), "vector": vector}
                        for (object_id, vector) in chunk
                    ],
                )

                progress.update(len(chunk))
            progress.close()

            # TODO: In the end, print a summary of how many objects have a feature vector now.
            stmt = (
                select([func.count()])
                .select_from(models.objects)
                .where(models.objects.c.vector.isnot(None))
            )
            n_initialized = conn.execute(stmt).scalar()

            stmt = select([func.count()]).select_from(models.objects)
            n_total = conn.execute(stmt).scalar()

            print(
                f"{n_initialized:,d} out of {n_total:,d} objects ({n_initialized/n_initialized:.2%}) now have a feature vector."
            )

            print("Done.")

    @app.cli.command()
    @click.argument("tree_fn")
    @click.argument("project_name", required=False, default=None)
    @click.option("--consolidate/--no-consolidate", default=True)
    def load_project(tree_fn, project_name, consolidate):
        """
        Load a project from a saved tree.
        """

        with database.engine.connect() as conn:
            tree = Tree(conn)

            if project_name is None:
                project_name = os.path.basename(os.path.splitext(tree_fn)[0])

            with conn.begin():
                print("Loading {}...".format(tree_fn))
                project_id = tree.load_project(project_name, tree_fn)
                root_id = tree.get_root_id(project_id)

                if consolidate:
                    print("Consolidating ...")
                    tree.consolidate_node(root_id)

            print("Root ID: {}".format(root_id))
            print("Project ID: {}".format(project_id))

    @app.cli.command()
    @click.argument("tree_fn")
    @click.argument("project_id")
    @click.option("--consolidate/--no-consolidate", default=True)
    @click.option("--with-offset/--without-offset", default=False)
    def update_project(tree_fn, project_id, consolidate, with_offset):
        """
        Load a project from a saved tree.
        """

        with database.engine.connect() as conn:
            tree = Tree(conn)

            with conn.begin():
                project = tree.get_project(project_id)
                project_str = f"{project['name']} ({project['project_id']})"

                if not click.confirm(f"Update {project_str} with {tree_fn}?"):
                    return

                print(f"Updating {project_str} with {tree_fn}...")
                saved_tree = processing.Tree.from_saved(tree_fn)

                # Apply offset
                if with_offset:
                    offset = (
                        tree.get_orig_node_id_offset(project["project_id"])
                        - saved_tree.nodes["node_id"].min()
                    )
                    saved_tree.offset_node_ids(offset)

                project_id = tree.update_project(project["project_id"], saved_tree)

                root_id = tree.get_root_id(project_id)

                if consolidate:
                    print("Consolidating ...")
                    tree.consolidate_node(root_id)

            print("Root ID: {}".format(root_id))
            print("Project ID: {}".format(project_id))

    @app.cli.command()
    @click.argument("root_id", type=int)
    @click.argument("tree_fn")
    def export_tree(root_id, tree_fn):
        """
        Export the whole tree with its objects.
        """
        with database.engine.connect() as conn:
            tree = Tree(conn)

            tree.export_tree(root_id, tree_fn)

    @app.cli.command()
    @click.argument("root_id", type=int, required=False)
    @click.option("--log/--no-log", "log", default=False)
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
    @click.argument("root_id", type=int)
    def connect_supertree(root_id):
        with database.engine.connect() as conn:
            tree = Tree(conn)
            tree.connect_supertree(root_id)

    def validate_consolidate_root_id(ctx, param, value):
        # We don't need these
        del ctx
        del param

        if value in ("all", "visible"):
            return value

        try:
            return int(value)
        except ValueError:
            raise click.BadParameter('root_id can be "all", "visible" or an actual id.')

    @app.cli.command()
    @click.argument("root_id", default="visible", callback=validate_consolidate_root_id)
    def consolidate(root_id):
        with database.engine.connect() as conn, Timer("Consolidate") as timer:
            tree = Tree(conn)

            if root_id == "all":
                print("Consolidating all projects...")
                root_ids = [p["node_id"] for p in tree.get_projects()]
            elif root_id == "visible":
                print("Consolidating visible projects...")
                root_ids = [p["node_id"] for p in tree.get_projects(True)]
            else:
                print("Consolidating {}...".format(root_id))
                root_ids = [root_id]

            for rid in root_ids:
                with timer.child(str(rid)):
                    print("Consolidating {}...".format(rid))
                    tree.consolidate_node(rid)
            print("Done.")

    @app.cli.command()
    @click.argument("project_id", type=int)
    def reset_grown(project_id: int):
        with database.engine.connect() as conn:
            tree = Tree(conn)

            tree.reset_grown(project_id)

            print("Done.")

    @app.cli.command()
    @click.argument("username")
    @click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
    def add_user(username, password):
        print("Adding user {}:".format(username))

        if not password:
            print("Password must not be empty!")
            return

        try:
            _add_user(username, password)
        except IntegrityError as e:
            print(e)
        else:
            print(f"User {username} added.")

    @app.cli.command()
    @click.argument("username")
    @click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
    def change_user(username, password):
        print("Changing user {}:".format(username))

        if not password:
            print("Password must not be empty!")
            return

        pwhash = generate_password_hash(
            password, method="pbkdf2:sha256:10000", salt_length=12
        )

        try:
            with database.engine.connect() as conn:
                stmt = models.users.update(
                    models.users.c.username == username, {"pwhash": pwhash}
                )
                conn.execute(stmt)
        except IntegrityError as e:
            print(e)
        else:
            print(f"User {username} changed.")

    @app.cli.command()
    @click.argument("project_id")
    @click.argument("labels_fn")
    @click.option("--clean-name/--no-clean-name", default=True)
    def export_labels(project_id, labels_fn, clean_name):
        with database.engine.connect() as conn:
            tree = Tree(conn)
            root_id = tree.get_root_id(project_id)
            processing_tree = tree.dump_tree(root_id)
            df = processing_tree.to_flat(clean_name=clean_name)
            df.to_csv(labels_fn, index=False)

    @app.cli.command()
    @click.argument("node_id")
    @click.argument("filename")
    def export_direct_objects(node_id, filename):
        with database.engine.connect() as conn, open(filename, "w") as f:
            tree = Tree(conn)

            f.writelines(
                "{}\n".format(o["object_id"]) for o in tree.get_objects(node_id)
            )

    @app.cli.command()
    @click.argument("filename")
    def export_log(filename):
        with database.engine.connect() as conn:
            log = pd.read_sql_query(
                select([models.log, models.nodes.c.project_id]).select_from(
                    models.log.outerjoin(models.nodes)
                ),
                conn,
                index_col="log_id",
            )
            log.to_csv(filename)

    @app.cli.command()
    def truncate_log():
        """
        Truncate the log.
        """
        print("Truncate log")
        print("WARNING: This is a destructive operation and all data will be lost.")

        if input("Continue? (y/n) ") != "y":
            print("Canceled.")
            return

        with database.engine.connect() as conn:
            stmt = models.log.delete()
            conn.execute(stmt)
