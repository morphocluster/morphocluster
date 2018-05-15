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

from morphocluster import database, models
from morphocluster.api import api
from morphocluster.database import engine
from morphocluster.models import objects
from morphocluster.tree import Tree

app = Flask(__name__)
app.config.from_object(__name__) # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.update(dict(
    SECRET_KEY='RuetVarbyap8',
    DATASET_PATH = "/data-ssd/mschroeder/NoveltyDetection/Dataset/",
    #DATASET_PATH = "/data1/mschroeder/Downloads/CIF/10_unstained_brightfield_10/",
    COLLECTION_FNS = ["collection_train.csv", "collection_unlabeled.csv"],
    REDIS_HOST = "localhost",
    REDIS_PORT = 6379,
    REDIS_DB = 0,
))

feature_fns = [
    "/data1/mschroeder/NoveltyDetection/Results/CrossVal/2018-01-26-11-54-41/n_features-32_split-0/collection_train_0_train.h5",
    "/data1/mschroeder/NoveltyDetection/Results/CrossVal/2018-01-26-11-54-41/n_features-32_split-0/collection_train_0_val.h5",
    "/data1/mschroeder/NoveltyDetection/Results/CrossVal/2018-01-26-11-54-41/n_features-32_split-0/collection_unlabeled.h5"
]

# Load projects
project_paths = ["/data1/mschroeder/NoveltyDetection/Results/CV-Clustering/2018-02-08-12-55-06/min_cluster_size-30_split-1",
                 "/data1/mschroeder/NoveltyDetection/Results/CV-Clustering/2018-02-08-12-55-06/min_cluster_size-20_split-0",
                 "/data1/mschroeder/NoveltyDetection/Results/CV-Clustering/2018-02-08-12-55-06/min_cluster_size-10_split-1",
                 "/data1/mschroeder/NoveltyDetection/Results/CV-Clustering/2018-02-08-12-55-06/min_cluster_size-5_split-1"]
project_paths = {os.path.basename(os.path.normpath(ppth)): ppth for ppth in project_paths}

PROJECT_ID = "min_cluster_size-20_split-0"

@app.cli.command()
@click.option('--drop/--no-drop', default=False)
def init_db(drop):
    print("Initializing the database.")
    
    with database.engine.begin() as txn:
        if drop and input("This is a destructive operation and all data will be lost. Continue? (y/n) ") == "y":
            database.metadata.drop_all(txn)
            
        database.metadata.create_all(txn)
    

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
@click.argument('features_fn')
def load_features(features_fn):
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
    with engine.connect() as conn:
        tree = Tree(conn)
        
        name = os.path.basename(os.path.normpath(project_path))
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
        with engine.connect() as conn:
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
        with engine.connect() as conn:
            stmt = models.users.insert({"username": username, "pwhash": pwhash})
            conn.execute(stmt)
    except IntegrityError as e:
        print(e)
    
    
@app.route("/")
def index():
    #===========================================================================
    # return render_template('pages/index.html',
    #                        projects = sorted(project_paths))
    #===========================================================================
    return redirect(url_for("labeling"))


@app.route("/labeling")
def labeling():
    return render_template('pages/labeling.html')


@app.route("/get_obj_image/<objid>")
def get_obj_image(objid):
    result = models.objects.select(models.objects.c.object_id == objid).execute().first()
    
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
    with engine.connect() as conn:
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
    
    if not auth or not check_auth(auth.username, auth.password):
        # Send a 401 response that enables basic auth
        return Response(
            'Could not verify your access level.\n'
            'You have to login with proper credentials', 401,
            {'WWW-Authenticate': 'Basic realm="Login Required"'})
