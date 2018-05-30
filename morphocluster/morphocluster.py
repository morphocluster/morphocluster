import itertools
import os
from getpass import getpass

import click
import h5py
import pandas as pd
from etaprogress.progress import ProgressBar
from flask import (Flask, Response, abort, redirect, render_template, request,
                   url_for)
from flask.helpers import send_from_directory
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.expression import bindparam
from werkzeug import check_password_hash, generate_password_hash

from morphocluster import models
from morphocluster.api import api
from morphocluster.models import objects, nodes, projects
from morphocluster.tree import Tree, CACHE_DEPTH_MAX
from time import sleep
from morphocluster.extensions import database, redis_store, migrate
import flask_migrate

app = Flask(__name__)

app.config.from_object('morphocluster.config_default')
app.config.from_envvar('MORPHOCLUSTER_SETTINGS')
app.config.from_object(__name__) # load config from this file , flaskr.py

# Initialize extensions
database.init_app(app)
redis_store.init_app(app)
migrate.init_app(app, database)

@app.cli.command()
def reset_db():
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
    with database.engine.begin() as txn:
        # Cached values are prefixed with an underscore
        cached_columns = list(c for c in nodes.columns.keys() if c.startswith("_"))
        values = {c: None for c in cached_columns}
        values["cache_depth"] = 0
        stmt = nodes.update().values(values)
        txn.execute(stmt)
        
    print("Cache was cleared.")
    
    
@app.cli.command()
def clear_projects():
    print("Clearing projects.")
    print("WARNING: This is a destructive operation and all data will be lost.")
    
    if input("Continue? (y/n) ") != "y":
        print("Canceled.")
        return
    
    print("Clearing project data...")
    with database.engine.begin() as txn:
        txn.execute(projects.delete())
        
@app.cli.command()
@click.option('--depth', type=int, default = CACHE_DEPTH_MAX)
def warm_cache(depth):
    print("Warming the cache (depth {})...".format(depth))
    with database.engine.connect() as conn:
        tree = Tree(conn)
        for p in tree.get_projects():
            tree.get_node(p["node_id"], cache_depth = depth)
    

@app.cli.command()
@click.argument('collection_fn')
def load_object_locations(collection_fn):
    # Load collections
    with database.engine.begin() as txn: 
        print("Loading {}...".format(collection_fn))
        data = pd.read_csv(collection_fn,
                           header = None,
                           names = ["object_id", "path", "label"],
                           usecols = ["object_id", "path"])    
         
        data_iter = data.itertuples()
        bar = ProgressBar(len(data), max_width=40)
        while True:
            chunk = tuple(itertools.islice(data_iter, 5000))
            if not chunk:
                break
            txn.execute(models.objects.insert(), [row._asdict() for row in chunk])
             
            bar.numerator += len(chunk)
            print(bar, end="\r")
        print()
        print("Done.")
            
            
@app.cli.command()
@click.argument('features_fns', nargs=-1)
def load_features(features_fns):
    for features_fn in features_fns:
        print("Loading {}...".format(features_fn))
        with h5py.File(features_fn, "r", libver="latest") as f_features, database.engine.begin() as txn:
            object_ids = f_features["objids"]
            vectors = f_features["features"]
            
            stmt = objects.update().where(objects.c.object_id == bindparam('_object_id')).values({
                'vector': bindparam('vector')
            })
            
            bar = ProgressBar(len(object_ids), max_width=40)
            obj_iter = iter(zip(object_ids, vectors))
            while True:
                chunk = tuple(itertools.islice(obj_iter, 1000))
                if not chunk:
                    break
                txn.execute(stmt, [{"_object_id": str(object_id), "vector": vector} for (object_id, vector) in chunk])
                  
                bar.numerator += len(chunk)
                print(bar, end="\r")
            print()
            print("Done.")
        
        
@app.cli.command()
@click.argument('project_path')
def load_project(project_path):
    with database.engine.connect() as conn:
        tree = Tree(conn)
        
        name = os.path.basename(os.path.normpath(project_path))
        
        with conn.begin():
            print("Loading...")
            project_id = tree.load_project(name, project_path)
            root_id = tree.get_root_id(project_id)
            print("Simplifying...")
            tree.flatten_tree(root_id)
            tree.prune_chains(root_id)
        
        
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
    
    pwhash = generate_password_hash(password, method='pbkdf2:sha256:10000', salt_length=12)
    
    try:
        with database.engine.connect() as conn:
            stmt = models.users.insert({"username": username, "pwhash": pwhash})
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
    
    pwhash = generate_password_hash(password, method='pbkdf2:sha256:10000', salt_length=12)
    
    try:
        with database.engine.connect() as conn:
            stmt = models.users.insert({"username": username, "pwhash": pwhash})
            conn.execute(stmt)
    except IntegrityError as e:
        print(e)
    
    
@app.route("/")
def index():
    return redirect(url_for("labeling"))


@app.route("/labeling")
def labeling():
    return render_template('pages/labeling.html')


@app.route("/get_obj_image/<objid>")
def get_obj_image(objid):
    with database.engine.connect() as conn:
        stmt = models.objects.select(models.objects.c.object_id == objid)
        result = conn.execute(stmt).first()
    
    if result is None:
        abort(404)
        
    return send_from_directory(app.config["DATASET_PATH"], result["path"])
    
    
# Register api
app.register_blueprint(api, url_prefix='/api')


#===============================================================================
# Authentication
#===============================================================================
def check_auth(username, password):
    # Retrieve entry from the database
    with database.engine.connect() as conn:
        stmt = models.users.select(models.users.c.username == username).limit(1)
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
