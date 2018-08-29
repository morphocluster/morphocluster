"""
Morphocluster: Main file.
"""
import itertools
import os
from getpass import getpass
from time import sleep

import click
import flask_migrate
import h5py
import pandas as pd
from etaprogress.progress import ProgressBar
from flask import (Flask, Response, abort, redirect, render_template, request,
                   url_for)
from flask.helpers import send_from_directory
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.expression import bindparam
from timer_cm import Timer
from werkzeug.security import check_password_hash, generate_password_hash

from morphocluster import models
from morphocluster.api import api
from morphocluster.extensions import database, migrate, redis_store
from morphocluster.tree import Tree

app = Flask(__name__)

app.config.from_object('morphocluster.config_default')
app.config.from_envvar('MORPHOCLUSTER_SETTINGS')

# Initialize extensions
database.init_app(app)
redis_store.init_app(app)
migrate.init_app(app, database)

# Enable batch mode
with app.app_context():
    database.engine.dialect.psycopg2_batch_mode = True


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
        affected_tables = [models.nodes, models.projects, models.nodes_objects]
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
@click.argument('project_path')
@click.argument('project_name', default=None)
@click.option('--root-last', "root_first", flag_value=False)
@click.option('--root-first', "root_first", flag_value=True, default=True)
def load_project(project_path, project_name, root_first):
    """
    Load a project from tree.csv and objids.csv.
    """

    with database.engine.connect() as conn:
        tree = Tree(conn)

        if project_name is None:
            project_name = os.path.basename(os.path.normpath(project_path))

        with conn.begin():
            print("Loading {} with root_first={!r}...".format(
                project_path, root_first))
            project_id = tree.load_project(
                project_name, project_path, root_first)
            tree.get_root_id(project_id)

@app.cli.command()
@click.argument('root_id', type=int)
@click.argument('fn_prefix')
def export_tree(root_id, fn_prefix):
    """
    Export the whole tree with its objects.
    """
    with database.engine.connect() as conn:
        tree = Tree(conn)

        tree.export_tree(root_id, fn_prefix)


@app.cli.command()
@click.argument('root_id', type=int)
def connect_supertree(root_id):
    with database.engine.connect() as conn:
        tree = Tree(conn)

        tree.connect_supertree(root_id)


@app.cli.command()
@click.argument('root_id', type=int)
def upgrade_nodes(root_id):
    with database.engine.connect() as conn:
        tree = Tree(conn)

        tree.upgrade_nodes(root_id)


@app.cli.command()
@click.argument('root_id', type=int)
def flatten_tree(root_id):
    with database.engine.connect() as conn:
        tree = Tree(conn)
        tree.flatten_tree(root_id)


@app.cli.command()
@click.argument('root_id', type=int)
def consolidate(root_id):
    with database.engine.connect() as conn, Timer("Consolidate"):
        tree = Tree(conn)
        tree.consolidate_node(root_id)


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
        log = pd.read_sql_query(models.log.select(), conn, index_col="log_id")
        log.to_csv(filename)


@app.route("/")
def index():
    return redirect(url_for("labeling"))


@app.route("/labeling")
def labeling():
    return render_template('pages/labeling.html')


@app.route("/bisect/<path:path>")
def vue(path):  # pylint: disable=unused-variable
    """
    Handle any routes handled by vue.
    """

    return app.send_static_file('vue/index.html')


@app.route("/get_obj_image/<objid>")
def get_obj_image(objid):
    with database.engine.connect() as conn:
        stmt = models.objects.select(models.objects.c.object_id == objid)
        result = conn.execute(stmt).first()

    if result is None:
        abort(404)

    response = send_from_directory(app.config["DATASET_PATH"], result["path"],
                                   conditional=True)

    response.headers['Cache-Control'] += ", immutable"

    return response


# Register api
app.register_blueprint(api, url_prefix='/api')


# ===============================================================================
# Authentication
# ===============================================================================
def check_auth(username, password):
    # Retrieve entry from the database
    with database.engine.connect() as conn:
        stmt = models.users.select(
            models.users.c.username == username).limit(1)
        user = conn.execute(stmt).first()

        if user is None:
            return False

    return check_password_hash(user["pwhash"], password)


@app.before_request
def require_auth():
    # exclude 404 errors and static routes
    # uses split to handle blueprint static routes as well
    if not request.endpoint or request.endpoint.rsplit('.', 1)[-1] == 'static':
        return

    auth = request.authorization

    success = check_auth(auth.username, auth.password) if auth else None

    if not auth or not success:
        if success is False:
            # Rate limiting for failed passwords
            sleep(1)

        # Send a 401 response that enables basic auth
        return Response(
            'Could not verify your access level.\n'
            'You have to login with proper credentials', 401,
            {'WWW-Authenticate': 'Basic realm="Login Required"'})
