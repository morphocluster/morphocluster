'''
Created on 19.03.2018

@author: mschroeder
'''
import datetime
import time
from distutils.util import strtobool

import numpy as np
from flask import jsonify as flask_jsonify, request
from flask.blueprints import Blueprint
from sklearn.manifold.isomap import Isomap

from morphocluster.tree import Tree, DEFAULT_CACHE_DEPTH
from urllib.parse import urlencode
import warnings
from morphocluster.classifier import Classifier
from functools import wraps
import json
from flask import Response
import uuid
import zlib
from redis.exceptions import RedisError
from morphocluster import models
from morphocluster.extensions import database, redis_store
from pprint import pprint
from flask.helpers import url_for
from itertools import chain
from flask_restful import reqparse


api = Blueprint("api", __name__)


def log(connection, action, node_id = None, reverse_action = None):
    auth = request.authorization
    username = auth.username if auth is not None else None
    
    stmt = models.log.insert({'node_id': node_id,
                              'username': username,
                              'action': action,
                              'reverse_action': reverse_action})
    
    connection.execute(stmt)


@api.record
def record(state):
    api.config = state.app.config

@api.after_request
def no_cache_header(response):
    response.headers['Last-Modified'] = datetime.datetime.now()
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def _node_icon(node):
    if node["starred"]:
        return "mdi mdi-star"
    
    if node["approved"]:
        return "mdi mdi-approval"
    
    return "mdi mdi-hexagon-multiple"

#===============================================================================
# /tree
#===============================================================================
def _tree_root(project):
    project["text"] = project["name"]
    project["children"] = True
    project["icon"] = "mdi mdi-tree"
    project["id"] = project["node_id"]
    
    return project

def _tree_node(node):
    result = {
        "id": node["node_id"],
        "text": "{} ({})".format(node["name"] or node["node_id"], node["n_children"]),
        "children": node["n_children"] > 0,
        "icon": _node_icon(node)
    }
    
    return result

@api.route("/tree", methods=["GET"])
def get_tree_root():
    with database.engine.connect() as connection:
        tree = Tree(connection)
        result = [_tree_root(p) for p in tree.get_projects()]
        
        return jsonify(result)
    
@api.route("/tree/<int:node_id>", methods=["GET"])
def get_subtree(node_id):
    with database.engine.connect() as connection:
        tree = Tree(connection)
        
        children = tree.get_children(node_id, order_by="n_children DESC")
        
        result = [_tree_node(c) for c in children]
        
        return jsonify(result)
    

#===============================================================================
# /projects
#===============================================================================



@api.route("/projects", methods=["GET"])
def get_projects():
    with database.engine.connect() as connection:
        tree = Tree(connection)
        return jsonify(tree.get_projects())
    
    
@api.route("/projects/<int:project_id>/", methods=["GET"])
def get_project(project_id):
    with database.engine.connect() as connection:
        tree = Tree(connection)
        log(connection, "get_project({})".format(project_id))
        return jsonify(_project(tree.get_project(project_id)))

#===============================================================================
# /nodes
#===============================================================================

@api.route("/nodes", methods=["POST"])
def create_node():
    """
    Create a new node.
    
    Request parameters:
        project_id
        name
        members
        starred
    """
    
    with database.engine.connect() as connection:
        tree = Tree(connection)
        data = request.get_json()
        
        object_ids = [m["object_id"] for m in data["members"] if "object_id" in m]
        node_ids = [m["node_id"] for m in data["members"] if "node_id" in m]
    
        project_id = data.get("project_id", None)
        name = data.get("name", None)
        parent_id = int(data.get("parent_id"))
        
        starred = strtobool(str(data.get("starred", "0")))
        
        if project_id is None:
            # Retrieve project_id for the parent_id
            project_id = tree.get_node(parent_id)["project_id"]
        
        print(data)
        
        node_id = tree.create_node(int(project_id), parent_id = parent_id, name = name, starred = starred)
        
        tree.relocate_nodes(node_ids, node_id)
        
        tree.relocate_objects(object_ids, node_id)
        
        log(connection, "create_node", node_id = node_id)
            
        node = tree.get_node(node_id, cache_depth=DEFAULT_CACHE_DEPTH)
          
        result = _node(tree, node)
        
        return jsonify(result)


def _node(tree, node, include_children=False):
    if node["name"] is None:
        node["name"] = node["node_id"]
    
    result = {
        "node_id": node["node_id"],
        "id": node["node_id"],
        "path": tree.get_path_ids(node["node_id"]),
        "text": "{} ({})".format(node["name"], node["n_children"]),
        "name": node["name"],
        "children": node["n_children"] > 0,
        "n_children": node["n_children"],
        "icon": _node_icon(node),
        "type_objects": node["_type_objects"],
        "starred": node["starred"],
        "approved": node["approved"],
        "own_type_objects": node["_own_type_objects"],
        "recursive_n_objects": int(node["_recursive_n_objects"]),
    }
    
    if include_children:
        result["children"] = [_node(tree, c) for c in tree.get_children(node["node_id"])]
    
    return result

def _object(object_):
    return {"object_id": object_["object_id"]}

#@api.route("/nodes/<int:node_id>/children", methods=["GET"])
def node_children(node_id):
    """
    Provide a collection of children for the given node.
    
    In the case of root, a singleton list is returned.
    
    This contains all fields required by jstree.
    
    Returns:
        List of dict:
            - id: node id
            - text: node name
            - children: true / false / list of child ids
            - icon: Icon for the node
            - n_children: number of children
    """
    with database.engine.connect() as connection:
        tree = Tree(connection)
        
        flags = {k: request.args.get(k, 0) for k in ("include_children")}
        expensive_values = DEFAULT_CACHE_DEPTH if flags["include_type_objects"] else 0
        
        result = [ _node(tree, c, **flags) for c in tree.get_children(node_id, expensive_values) ]
            
        return jsonify(result)

#@api.route("/nodes/<int:node_id>/leaves", methods=["GET"])
def get_node_leaves(node_id):
    """
    Provide a collection of leaves for the given node.
    
    Data is in the same format as node_children.
    
    Returns:
        List of dict:
            - id: node id
            - text: node name
            - children: true / false / list of child ids
            - icon: Icon for the node
            - n_children: number of children
    """
    flags = {k: request.args.get(k, 0) for k in ("include_preview",)}
    
    result = [ _node(tree, c, **flags) for c in tree.get_leaves(node_id) ]
        
    return jsonify(result)

#@api.route("/nodes/<int:node_id>/objects", methods=["GET"])
def get_node_objects(node_id):
    """
    Provide a collection of objids for the given node.
    
    Returns:
        List of objids
    """
    with database.engine.connect() as connection:
        tree = Tree(connection)
    
        objects = tree.get_objects(node_id)
        
        slice_ = slice(request.args.get("start", None, type=int),
                       request.args.get("stop", None, type=int))
        
        return jsonify([o["object_id"] for o in objects[slice_]])


def _arrange_by_sim(result):
    """
    Return empty tuple for unchanged order.
    """
    ISOMAP_FIT_SUBSAMPLE_N = 1000
    ISOMAP_N_NEIGHBORS = 5
    
    if len(result) <= ISOMAP_N_NEIGHBORS:
        return ()
    
    # Get vector values
    vectors = np.array([ m["_centroid"] if "_centroid" in m else m["vector"] for m in result ],
                       dtype = float)
            
    print("Arranging {:,d} elements by similarity...".format(len(vectors)))
    
    if vectors.shape[0] <= ISOMAP_FIT_SUBSAMPLE_N:
        subsample = vectors
    else:
        idxs = np.random.choice(vectors.shape[0], ISOMAP_FIT_SUBSAMPLE_N, replace=False)
        subsample = vectors[idxs]
    
    start = time.perf_counter()
    isomap = Isomap(n_components=1, n_neighbors=ISOMAP_N_NEIGHBORS, n_jobs=4).fit(subsample)
    order = np.squeeze(isomap.transform(vectors))
    elapsed = time.perf_counter() - start
    
    print("Arranging of {:,d} elements took {}s".format(len(vectors), elapsed))
    
    order = np.argsort(order)
        
    return order


def _arrange_by_nleaves(result):
    n_leaves = np.array([ len(m["_leaves"]) if "_leaves" in m else 0 for m in result ],
                        dtype = int)
    
    return np.argsort(n_leaves)[::-1]


def _members(tree, members):
    return [_node(tree, m) if "node_id" in m else _object(m) for m in members]

def batch(iterable, n=1):
    """
    Taken from https://stackoverflow.com/a/8290508/1116842
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def json_dumps(o, *args, **kwargs):
    try:
        return json.dumps(o, *args, **kwargs)
    except TypeError:
        pprint(o)
        raise
    
def jsonify(*args, **kwargs):
    try:
        return flask_jsonify(*args, **kwargs)
    except TypeError:
        pprint(args)
        pprint(kwargs)
        raise


def cache_serialize_page(func, page_size = 100, compress = True):
    """
    `func` is expected to return a json-serializable list.
    It gains the `page` and `request_id` parameter. The resulting list is split into batches of `page_size` items.
    
    Parameters:
        func: func(*args, **kwargs) -> list
        
    Return:
        (page, n_pages):
            page: `json.dumps`ed `page`-th batch of result of the call to `func`.
            n_pages: Total number of pages
    
    Example:
        @cache_serialize_page
        def foo():
            return ["a", "b", "c"]
            
        foo(page=0) -> "a", True
    """
    @wraps(func)
    def wrapper(*args, page = None, request_id = None, **kwargs):
        if page is None:
            raise ValueError("page may not be None!")
        
        cache_key = '{}:{}:{}'.format(func.__name__,
                                      json_dumps([args, kwargs], sort_keys = True, separators=(',', ':')),
                                      request_id)
        
        try:
            page_result = redis_store.lindex(cache_key, page)
        
            if page_result is not None:
                n_pages = redis_store.llen(cache_key)
                
                if compress:
                    page_result = zlib.decompress(page_result)
                    
                #print("Returning page {} from cached result".format(page))
                
                return page_result, n_pages
            
        except RedisError as e:
            warnings.warn("RedisError: {}".format(e))
            
        # Calculate result
        result = func(*args, **kwargs)
        
        # Paginate full_result
        pages = batch(result, page_size)
        
        # Serialize individual pages
        pages = [json_dumps(p) for p in pages]
        
        n_pages = len(pages)
        
        if n_pages:
            if compress:
                #raw_length = sum(len(p) for p in pages)
                cache_pages = [zlib.compress(p.encode()) for p in pages]
                #compressed_length = sum(len(p) for p in pages)
                
                #print("Compressed pages. Ratio: {:.2%}".format(compressed_length / raw_length))
            else:
                cache_pages = pages
                
            try:
                redis_store.rpush(cache_key, *cache_pages)
            except RedisError as e:
                warnings.warn("RedisError: {}".format(e))
        
        if 0 <= page < n_pages:
            return pages[page], n_pages
        
        return "[]", n_pages
    
    return wrapper
    
def seq2array(seq, dtype, length):
    """
    Converts a sequence consisting of `numpy array`s to a single array.
    Elements that are None are converted to an appropriate zero entry.
    """
    
    seq = iter(seq)
    leading = []
    zero = None
    
    for x in seq:
        leading.append(x)
        if x is not None:
            zero = np.zeros_like(x)
            break
        
    if zero is None:
        raise ValueError("Empty sequence or only None")
    
    array = np.empty((length,) + zero.shape, zero.dtype)
    for i, x in enumerate(chain(leading, seq)):
        array[i] = zero if x is None else x 
    
    return array
    
def _arrange_by_starred_sim(result, starred):
    if len(starred) == 0:
        return _arrange_by_sim(result)
    
    if len(result) == 0:
        return ()
    
    # Get vectors
    vectors = seq2array((m["_centroid"] if "_centroid" in m else m["vector"] for m in result),
                        float, len(result))
    starred_vectors = seq2array((m["_centroid"] for m in starred), float, len(starred))

    try:
        classifier = Classifier(starred_vectors)
        distances = classifier.distances(vectors)
        max_dist = np.max(distances, axis=0)
        max_dist_idx = np.argsort(max_dist)[::-1]
        
        assert len(max_dist_idx) == len(result), "{} != {}".format(len(max_dist_idx), len(result))
        
        return max_dist_idx
        
    except:
        print("starred_vectors", starred_vectors.shape)
        print("vectors", vectors.shape)
        raise


@cache_serialize_page
def _get_node_members(node_id, nodes = False, objects = False, arrange_by = "", starred_first = False):
    with database.engine.connect() as connection:
        tree = Tree(connection)
        
        sorted_nodes_include = "unstarred" if starred_first else None
        
        result = []
        if nodes:
            result.extend(tree.get_children(node_id, cache_depth = DEFAULT_CACHE_DEPTH, include=sorted_nodes_include))
        if objects:
            result.extend(tree.get_objects(node_id))
            
        if arrange_by == "starred_sim" or starred_first:
            starred = tree.get_children(node_id, cache_depth = DEFAULT_CACHE_DEPTH, include="starred")
            
        if arrange_by != "":
            result = np.array(result, dtype=object)
            
            if arrange_by == "sim":
                order = _arrange_by_sim(result)
            elif arrange_by == "nleaves":
                order = _arrange_by_nleaves(result)
            elif arrange_by == "starred_sim":
                starred = tree.get_children(node_id, cache_depth = DEFAULT_CACHE_DEPTH, include="starred")
                order = _arrange_by_starred_sim(result, starred)
            else:
                warnings.warn("arrange_by={} not supported!".format(arrange_by))
                order = ()
            
            result = result[order].tolist()
            
        if starred_first:
            result = starred + result
            
        result = _members(tree, result)
    
        return result


@api.route("/nodes/<int:node_id>/members", methods=["GET"])
def get_node_members(node_id):
    """
    Provide a collection of objects and/or children.
    
    Parameters:
        node_id (int): ID of a node
        
    Request parameters:
        nodes (boolean): Include nodes in the response?
        objects (boolean): Include objects in the response?
        arrange_by ("sim"|"nleaves"): Arrange members by similarity / number of leaves / ...
        page (int): Page number (default 0)
        request_id (str, optional): Identification string for the current request collection.
        starred_first (boolean): Return starred children first (default: 0)
    
    Returns:
        List of members
    """
    
    parser = reqparse.RequestParser()
    parser.add_argument("nodes", type=strtobool, default = 0)
    parser.add_argument("objects", type=strtobool, default = 0)
    parser.add_argument("arrange_by", default = "")
    parser.add_argument("page", type = int, default = 0)
    parser.add_argument("request_id")
    parser.add_argument("starred_first", type=strtobool, default = 1)
    
    arguments = parser.parse_args(strict=True)
    
    if arguments.request_id is None:
        arguments.request_id = uuid.uuid4().hex
    
    result, n_pages = _get_node_members(node_id, arguments.nodes, arguments.objects, arguments.arrange_by,
                                        arguments.starred_first,
                                        page = arguments.page,
                                        request_id = arguments.request_id)
        
    # Generate response
    response = Response(result, mimetype=api.config['JSONIFY_MIMETYPE'])

    #=======================================================================
    # Generate Link response header
    #=======================================================================
    link_header_fields = []
    link_parameters = dict(arguments)
    if 0 < arguments.page < n_pages:
        # Link to previous page
        link_parameters["page"] = arguments.page - 1
        url = "{}?{}".format(url_for(".get_node_members", node_id = node_id), urlencode(link_parameters))
        link_header_fields.append('<{}>; rel="previous"'.format(url))
    
    
    if arguments.page + 1 < n_pages:
        # Link to next page
        link_parameters["page"] = arguments.page + 1
        url = "{}?{}".format(url_for(".get_node_members", node_id = node_id), urlencode(link_parameters))
        link_header_fields.append('<{}>; rel="next"'.format(url))
        
    # Link to last page
    link_parameters["page"] = n_pages - 1
    url = "{}?{}".format(url_for(".get_node_members", node_id = node_id), urlencode(link_parameters))
    link_header_fields.append('<{}>; rel="last"'.format(url))
    
    response.headers["Link"] = ",". join(link_header_fields)
                
    return response

@api.route("/nodes/<int:node_id>/members", methods=["POST"])
def post_node_members(node_id):
    data = request.get_json()
    
    object_ids = [d["object_id"] for d in data if "object_id" in d]
    node_ids = [d["node_id"] for d in data if "node_id" in d]
    
    print("new nodes:", node_ids)
    print("new objects:", object_ids)
    
    with database.engine.connect() as connection:
        tree = Tree(connection)

        with connection.begin():
            tree.relocate_nodes(node_ids, node_id)
            tree.relocate_objects(object_ids, node_id)
    
    return jsonify("ok")


@api.route("/nodes/<int:node_id>", methods=["GET"])
def get_node(node_id):    
    with database.engine.connect() as connection:
        tree = Tree(connection)
        
        flags = {k: request.args.get(k, 0) for k in ("include_children",)}
        
        node = tree.get_node(node_id, cache_depth=DEFAULT_CACHE_DEPTH)
        
        log(connection, "get_node", node_id = node_id)
        
        result = _node(tree, node, **flags)
        
        return jsonify(result)
    
    
@api.route("/nodes/<int:node_id>", methods=["PATCH"])
def patch_node(node_id):
    with database.engine.connect() as connection:
        tree = Tree(connection)
        
        data = request.get_json()
        flags = {k: request.args.get(k, 0) for k in ("include_children",)}
        
        # TODO: Use argparse
        if "starred" in data:
            data["starred"] = strtobool(str(data["starred"]))
            
        if "parent_id" in data:
            raise ValueError("parent_id must not be set directly, use /nodes/<node_id>/adopt.")
        
        tree.update_node(node_id, data)
        
        log(connection, "update_node", node_id = node_id)
        
        node = tree.get_node(node_id, True)
        
        result = _node(tree, node, **flags)
        
        return jsonify(result)
    
@api.route("/nodes/<int:parent_id>/adopt_members", methods=["POST"])
def node_adopt_members(parent_id):
    """
    Adopt a list of nodes.
    
    Parameters:
        parent_id (int): ID of the node that accepts new members.
        
    Request data:
        members: List of nodes ({node_id: ...}) and objects ({object_id: ...}).
    
    Returns:
        Nothing.
    """
    with database.engine.connect() as connection:
        tree = Tree(connection)
        
        members = request.get_json()["members"]
        
        node_ids = [int(m["node_id"]) for m in members if "node_id" in m]
        object_ids = [m["object_id"] for m in members if "object_id" in m]
        
        tree.relocate_nodes(node_ids, parent_id)
        tree.relocate_objects(object_ids, parent_id)
        
        return jsonify({})
        

@api.route("/nodes/<int:node_id>/recommended_children", methods=["GET"])
def node_get_recommended_children(node_id):
    with database.engine.connect() as connection:
        tree = Tree(connection)
    
        flags = {k: request.args.get(k, 0) for k in ("include_children",)}
        result = [ _node(tree, c, **flags) for c in tree.recommend_children(node_id) ]
        
        return jsonify(result)

@api.route("/nodes/<int:node_id>/recommended_objects", methods=["GET"])
def node_get_recommended_objects(node_id):
    with database.engine.connect() as connection:
        tree = Tree(connection)
    
        result = [ _object(o) for o in tree.recommend_objects(node_id) ]
        
        return jsonify(result)
    
@api.route("/nodes/<int:node_id>/tip", methods=["GET"])
def node_get_tip(node_id):
    with database.engine.connect() as connection:
        tree = Tree(connection)
    
        return jsonify(tree.get_tip(node_id))
    
    
@api.route("/nodes/<int:node_id>/n_sorted", methods=["GET"])
def node_get_n_sorted(node_id):
    with database.engine.connect() as connection:
        tree = Tree(connection)
        
        nodes = tree.get_minlevel_starred(node_id, cache_depth = DEFAULT_CACHE_DEPTH)
        
        n_sorted = sum(n["_recursive_n_objects"] for n in nodes)
        
        return jsonify(n_sorted)
    
    
@api.route("/nodes/<int:node_id>/merge_into", methods=["POST"])
def post_node_merge_into(node_id):
    """
    Merge a node into another node.
    
    Parameters:
        node_id: Node that is merged.
        
    Request parameters:
        dest_node_id: Node that absorbs the children and objects.
    """
    with database.engine.connect() as connection:
        tree = Tree(connection)
        
        data = request.get_json()
        
        tree.merge_node_into(node_id, data["dest_node_id"])
        
        log(connection, "merge_node_into({}, {})".format(node_id, data["dest_node_id"]),
            node_id = data["dest_node_id"])
        
        return jsonify(None)
    
@api.route("/nodes/<int:node_id>/classify", methods=["POST"])
def post_node_classify(node_id):
    """
    Classify the members of a node into their starred siblings.
    
    Parameters:
        node_id: Parent of the classified members.
    """
    with database.engine.connect() as connection:
        tree = Tree(connection)
        
        # Split children into starred and unstarred
        with connection.begin():
            children = tree.get_children(node_id, cache_depth = DEFAULT_CACHE_DEPTH)
            
            starred = []
            unstarred = []
            for c in children:
                (starred if c["starred"] else unstarred).append(c)
                
            starred_centroids = np.array([c["_centroid"] for c in starred])
            unstarred_centroids = np.array([c["_centroid"] for c in unstarred])
            unstarred_ids = np.array([c["node_id"] for c in unstarred])
            
            classifier = Classifier(starred_centroids)
            
            # Predict unstarred children (if any)
            n_unstarred = len(unstarred_centroids)
            if n_unstarred > 0:
                print("Predicting {} unstarred children of {}...".format(n_unstarred, node_id))
                type_predicted = classifier.classify(unstarred_centroids)
                
                print(type_predicted)
                
                for i, starred_node in enumerate(starred):
                    nodes_to_move = [int(n) for n in unstarred_ids[type_predicted == i]]
                    tree.relocate_nodes(nodes_to_move, starred_node["node_id"])
                    
                n_predicted_children = np.sum(type_predicted > -1)
            else:
                n_predicted_children = 0
            
            
            #Predict objects
            objects = tree.get_objects(node_id)
            print("Predicting {} objects of {}...".format(len(objects), node_id))
            object_vectors = np.array([o["vector"] for o in objects])
            object_ids = np.array([o["object_id"] for o in objects])
            
            type_predicted = classifier.classify(object_vectors)
            
            for i, starred_node in enumerate(starred):
                objects_to_move = [str(o) for o in object_ids[type_predicted == i]]
                tree.relocate_objects(objects_to_move, starred_node["node_id"])
                
            n_predicted_objects = np.sum(type_predicted > -1)
            
            log(connection, "classify_members", node_id = node_id)
            
            return jsonify({"n_predicted_children": int(n_predicted_children),
                            "n_predicted_objects": int(n_predicted_objects)})
    
