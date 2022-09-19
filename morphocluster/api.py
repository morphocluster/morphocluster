"""
Created on 19.03.2018

@author: mschroeder
"""
import json
import os
import traceback
import uuid
import warnings
import zlib
from datetime import datetime
from distutils.util import strtobool
from functools import wraps
from pprint import pprint

import numpy as np
import pandas as pd
import werkzeug.exceptions
from flask import Response
from flask import current_app as app
from flask import jsonify as flask_jsonify
from flask import request
from flask.blueprints import Blueprint
from flask.helpers import url_for
from flask_restful import reqparse
from redis.exceptions import RedisError
from sklearn.manifold import Isomap
from timer_cm import Timer

from morphocluster import background, models
from morphocluster.classifier import Classifier
from morphocluster.extensions import database, redis_lru, rq
from morphocluster.helpers import keydefaultdict, seq2array
from morphocluster.schemas import JobSchema, LogSchema
from morphocluster.tree import Tree

api = Blueprint("api", __name__)

from werkzeug.exceptions import HTTPException


def _complex2repr(o):
    if isinstance(o, (int, str, float)):
        return o

    if isinstance(o, (list, tuple)):
        return [_complex2repr(v) for v in o]

    return repr(o)


@api.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""

    data = {
        "code": e.code,
        "name": e.name,
        "description": e.description,
    }

    # flask_restful.abort populates the data attribute
    data.update(getattr(e, "data", {}))

    # Store traceback
    data["traceback"] = traceback.format_exc()

    # Convert all complex values to their representation
    data = {k: _complex2repr(v) for k, v in data.items()}

    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps(data)
    response.content_type = "application/json"
    return response


def batch(iterable, n=1):
    """
    Taken from https://stackoverflow.com/a/8290508/1116842
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def json_converter(value):
    if isinstance(value, np.floating):  # type: ignore
        return float(value)
    if isinstance(value, np.integer):  # type: ignore
        return int(value)
    if isinstance(value, np.bool):
        return bool(value)
    raise TypeError("Unknown type: {!r}".format(type(value)))


def json_dumps(o, *args, **kwargs):
    try:
        return json.dumps(o, *args, **kwargs, default=json_converter)
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


def log(connection, action, node_id=None, reverse_action=None, data=None):
    auth = request.authorization
    username = auth.username if auth is not None else None  # type: ignore

    stmt = models.log.insert(
        {
            "node_id": node_id,
            "username": username,
            "action": action,
            "reverse_action": reverse_action,
            "data": data,
        }
    )

    connection.execute(stmt)


@api.record
def record(state):
    api.config = state.app.config  # type: ignore


@api.after_request
def api_headers(response):
    response.headers["Last-Modified"] = datetime.now()
    response.headers[
        "Cache-Control"
    ] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"

    return response


def _node_icon(node):
    if node["starred"]:
        return "mdi mdi-star"

    if node["approved"]:
        return "mdi mdi-check-decagram"

    return "mdi mdi-hexagon-multiple"


# ===============================================================================
# /tree
# ===============================================================================


def _tree_root(project):
    project["text"] = project["name"]
    project["children"] = True
    project["icon"] = "mdi mdi-tree"
    project["id"] = project["node_id"]

    return project


def _tree_node(node, supertree=False):
    result = {
        "id": node["node_id"],
        "text": "{} ({})".format(node["name"] or node["node_id"], node["_n_children"]),
        "children": node["n_superchildren"] > 0
        if supertree
        else node["_n_children"] > 0,
        "icon": _node_icon(node),
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
    flags = {k: request.args.get(k, 0, strtobool) for k in ("supertree",)}  # type: ignore

    with database.engine.connect() as connection:
        tree = Tree(connection)

        if flags["supertree"]:
            children = tree.get_children(
                node_id, supertree=True, include="starred", order_by="_n_children DESC"
            )
        else:
            children = tree.get_children(node_id, order_by="_n_children DESC")

        result = [_tree_node(c, flags["supertree"]) for c in children]

        return jsonify(result)


# ===============================================================================
# /projects
# ===============================================================================
@api.route("/projects", methods=["GET"])
def get_projects():

    parser = reqparse.RequestParser()
    parser.add_argument("include_progress", type=strtobool, default=0)
    arguments = parser.parse_args(strict=True)

    with database.engine.connect() as connection:
        tree = Tree(connection)

        projects = tree.get_projects()

        if arguments["include_progress"]:
            for p in projects:
                progress = tree.calculate_progress(p["node_id"])
                p["progress"] = progress

        return jsonify(projects)


@api.route("/projects/<int:project_id>", methods=["GET"])
def get_project(project_id):

    parser = reqparse.RequestParser()
    parser.add_argument("include_progress", type=strtobool, default=0)
    arguments = parser.parse_args(strict=True)

    with database.engine.connect() as connection:
        tree = Tree(connection)
        result = tree.get_project(project_id)

        if arguments["include_progress"]:
            progress = tree.calculate_progress(result["node_id"])
            result["progress"] = progress

        return jsonify(result)


@api.route("/projects/<int:project_id>/unfilled_nodes", methods=["GET"])
def get_unfilled_nodes(project_id):
    # with database.engine.connect() as connection:
    #     tree = Tree(connection)
    #     result = tree.get_project(project_id)

    #     if arguments["include_progress"]:
    #         progress = tree.calculate_progress(result["node_id"])
    #         result["progress"] = progress

    #     return jsonify(result)

    # TODO
    return jsonify([10622])


@api.route("/projects/<int:project_id>/save", methods=["POST"])
def save_project(project_id):
    """
    Save the project at PROJECT_EXPORT_DIR.
    """
    with database.engine.connect() as conn:
        tree = Tree(conn)

        project = tree.get_project(project_id)

        root_id = tree.get_root_id(project_id)

        tree_fn = os.path.join(
            api.config["PROJECT_EXPORT_DIR"],  # type: ignore
            "{:%Y-%m-%d-%H-%M-%S}--{}--{}.zip".format(
                datetime.now(), project["project_id"], project["name"]
            ),
        )

        tree.export_tree(root_id, tree_fn)

        return jsonify({"tree_fn": tree_fn})


# ===============================================================================
# /nodes
# ===============================================================================


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
        data = request.get_json()  # type: ignore

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

        with connection.begin():
            node_id = tree.create_node(
                int(project_id), parent_id=parent_id, name=name, starred=starred
            )

            tree.relocate_nodes(node_ids, node_id)

            tree.relocate_objects(object_ids, node_id)

            log(connection, "create_node", node_id=node_id)

            node = tree.get_node(node_id, require_valid=True)

            print("Created node {}.".format(node_id))

        result = _node(tree, node)

        return jsonify(result)


def _node(tree, node, include_children=False):
    if node["name"] is None:
        node["name"] = node["node_id"]

    result = {
        "node_id": node["node_id"],
        "id": node["node_id"],
        "path": tree.get_path_ids(node["node_id"]),
        "text": "{} ({})".format(node["name"], node["_n_children"]),
        "name": node["name"],
        "children": node["_n_children"] > 0,
        "n_children": node["_n_children"],
        "icon": _node_icon(node),
        "type_objects": node["_type_objects"],
        "starred": node["starred"],
        "approved": node["approved"],
        "own_type_objects": node["_own_type_objects"],
        "n_objects_deep": node["_n_objects_deep"] or 0,
        "n_objects": node["_n_objects"] or 0,
        "parent_id": node["parent_id"],
        "project_id": node["project_id"],
        "filled": node["filled"],
    }

    if include_children:
        result["children"] = [
            _node(tree, c) for c in tree.get_children(node["node_id"])
        ]

    return result


def _object(object_):
    return {"object_id": object_["object_id"]}


def _arrange_by_sim(result):
    """
    Return empty tuple for unchanged order.
    """
    ISOMAP_FIT_SUBSAMPLE_N = 1000
    ISOMAP_N_NEIGHBORS = 5

    if len(result) <= ISOMAP_N_NEIGHBORS:
        return ()

    # Get vector values
    vectors = seq2array(
        [m["_centroid"] if "_centroid" in m else m["vector"] for m in result],
        len(result),
    )

    if vectors.shape[0] <= ISOMAP_FIT_SUBSAMPLE_N:
        subsample = vectors
    else:
        idxs = np.random.choice(vectors.shape[0], ISOMAP_FIT_SUBSAMPLE_N, replace=False)
        subsample = vectors[idxs]

    try:
        isomap = Isomap(n_components=1, n_neighbors=ISOMAP_N_NEIGHBORS, n_jobs=4).fit(
            subsample
        )
        order = np.squeeze(isomap.transform(vectors))
    except ValueError:
        print(subsample)
        raise

    order = np.argsort(order)

    return order


def _arrange_by_nleaves(result):
    n_leaves = np.array(
        [len(m["_leaves"]) if "_leaves" in m else 0 for m in result], dtype=int
    )

    return np.argsort(n_leaves)[::-1]


def _members(tree, members):
    return [_node(tree, m) if "node_id" in m else _object(m) for m in members]


def _load_or_calc(func, func_kwargs, request_id, page, page_size=100, compress=True):
    print("Load or calc {}...".format(func.__name__))

    # If a request_id is given, load the result from the cache
    if request_id is not None:
        cache_key = "{}:{}".format(func.__name__, request_id)
        try:
            print("Loading cache key {}...".format(cache_key))
            page_result = redis_lru.lindex(cache_key, page)

            if page_result is None:
                raise ValueError("Unknown cache_key: {}".format(cache_key))

            n_pages = redis_lru.llen(cache_key)

            if compress:
                page_result = zlib.decompress(page_result).decode()

            # print("Returning page {} from cached result".format(page))

            return page_result, n_pages, request_id

        except RedisError as exc:
            raise ValueError(
                "Could not retrieve cache_key: {}".format(cache_key)
            ) from exc

    # Otherwise calculate a result
    request_id = uuid.uuid4().hex
    cache_key = "{}:{}".format(func.__name__, request_id)

    # Calculate result
    result = func(**func_kwargs)

    # Paginate full_result
    pages = batch(result, page_size)

    # Serialize individual pages
    pages = [json_dumps(p) for p in pages]

    n_pages = len(pages)

    if n_pages:
        if compress:
            # raw_length = sum(len(p) for p in pages)
            cache_pages = [zlib.compress(p.encode()) for p in pages]
            # compressed_length = sum(len(p) for p in pages)

            # print("Compressed pages. Ratio: {:.2%}".format(compressed_length / raw_length))
        else:
            cache_pages = pages

        try:
            redis_lru.rpush(cache_key, *cache_pages)
        except RedisError as e:
            warnings.warn("RedisError: {}".format(e))

    if 0 <= page < n_pages:
        return pages[page], n_pages, request_id

    return "[]", n_pages, request_id


def cache_serialize_page(endpoint, **kwargs):
    """
    `func` is expected to return a json-serializable list.
    It gains the `page` and `request_id` parameter. The resulting list is split into batches of `page_size` items.

    Decorated Function:
        func: func(**kwargs) -> list

        ! func is expected to only take keyword parameters !

    Return:
        Response()

    Example:
        @cache_serialize_page()
        def foo():
            return ["a", "b", "c"]

        foo(page=0) -> "a", True
    """

    def decorator(func):
        @wraps(func)
        def wrapper(page=None, request_id=None, **func_kwargs):
            if page is None:
                raise ValueError("page may not be None!")

            raw_result, n_pages, request_id = _load_or_calc(
                func, func_kwargs, request_id, page, **kwargs
            )

            meta = {
                "request_id": request_id,
                "last_page": n_pages - 1,
            }

            if 0 < page < n_pages:
                meta["previous_page"] = page - 1

            if page + 1 < n_pages:
                meta["next_page"] = page + 1

            link_parameters = func_kwargs.copy()
            link_parameters["request_id"] = request_id
            links = {"self": url_for(endpoint, **link_parameters)}

            result = {"meta": meta, "links": links, "data": "$data-placeholder$"}

            result = json_dumps(result)

            result = result.replace('"$data-placeholder$"', raw_result)

            # ===================================================================
            # Construct response
            # ===================================================================
            response = Response(result, mimetype=api.config["JSONIFY_MIMETYPE"])  # type: ignore

            # =======================================================================
            # Generate Link response header
            # =======================================================================
            link_header_fields = []

            if 0 < page < n_pages:
                # Link to previous page
                link_parameters["page"] = page - 1
                url = url_for(endpoint, **link_parameters)
                link_header_fields.append('<{}>; rel="previous"'.format(url))

            if page + 1 < n_pages:
                # Link to next page
                link_parameters["page"] = page + 1
                url = url_for(endpoint, **link_parameters)
                link_header_fields.append('<{}>; rel="next"'.format(url))

            # Link to last page
            link_parameters["page"] = n_pages - 1
            url = url_for(endpoint, **link_parameters)
            link_header_fields.append('<{}>; rel="last"'.format(url))

            response.headers["Link"] = ",".join(link_header_fields)

            return response

        return wrapper

    return decorator


def _arrange_by_starred_sim(result, starred):
    if len(starred) == 0:
        return _arrange_by_sim(result)

    if len(result) == 0:
        return ()

    try:
        # Get vectors
        vectors = seq2array(
            (m["_centroid"] if "_centroid" in m else m["vector"] for m in result),
            len(result),
        )
        starred_vectors = seq2array((m["_centroid"] for m in starred), len(starred))
    except ValueError as e:
        print(e)
        return ()

    try:
        classifier = Classifier(starred_vectors)
        distances = classifier.distances(vectors)
        max_dist = np.max(distances, axis=0)
        max_dist_idx = np.argsort(max_dist)[::-1]

        assert len(max_dist_idx) == len(result), "{} != {}".format(
            len(max_dist_idx), len(result)
        )

        return max_dist_idx

    except:
        print("starred_vectors", starred_vectors.shape)
        print("vectors", vectors.shape)
        raise


@cache_serialize_page(".get_node_members")
def _get_node_members(
    node_id,
    nodes=False,
    objects=False,
    arrange_by="",
    starred_first=False,
    descending=False,
):
    with database.engine.connect() as connection, Timer("_get_node_members") as timer:
        tree = Tree(connection)

        sorted_nodes_include = "unstarred" if starred_first else None

        result = []
        if nodes:
            with timer.child("tree.get_children()"):
                result.extend(tree.get_children(node_id, include=sorted_nodes_include))
        if objects:
            with timer.child("tree.get_objects()"):
                result.extend(tree.get_objects(node_id))

        if arrange_by == "starred_sim" or starred_first:
            with timer.child("tree.get_children(starred)"):
                starred = tree.get_children(node_id, include="starred")

        if arrange_by != "":
            result = np.array(result, dtype=object)

            if arrange_by == "sim":
                with timer.child("sim"):
                    order = _arrange_by_sim(result)
            elif arrange_by == "nleaves":
                with timer.child("nleaves"):
                    order = _arrange_by_nleaves(result)
            elif arrange_by == "starred_sim":
                with timer.child("starred_sim"):
                    # If no starred members yet, arrange by distance to regular children
                    anchors = starred if len(starred) else tree.get_children(node_id)

                    order = _arrange_by_starred_sim(result, anchors)
            elif arrange_by == "interleaved":
                with timer.child("interleaved"):
                    order = _arrange_by_sim(result)
                    if len(order):
                        order0, order1 = np.array_split(order.copy(), 2)
                        order[::2] = order0
                        order[1::2] = order1[::-1]
            elif arrange_by == "random":
                with timer.child("random"):
                    order = np.random.permutation(len(result))
            else:
                warnings.warn("arrange_by={} not supported!".format(arrange_by))
                order = ()

            if descending:
                order = order[::-1]

            # ===================================================================
            # if len(order):
            #     try:
            #         assert np.all(np.bincount(order) == 1)
            #     except:
            #         print(order)
            #         print(np.bincount(order))
            #         raise
            # ===================================================================

            result = result[order].tolist()

        if starred_first:
            result = starred + result

        result = _members(tree, result)

        return result


@api.route("/nodes/<int:node_id>/members", methods=["GET"])
def get_node_members(node_id):
    """
    Provide a collection of objects and/or children.

    URL parameters:
        node_id (int): ID of a node

    Request parameters:
        nodes (boolean): Include nodes in the response?
        objects (boolean): Include objects in the response?
        arrange_by ("sim"|"nleaves"): Arrange members by similarity / number of leaves / ...
        page (int): Page number (default 0)
        request_id (str, optional): Identification string for the current request collection.
        starred_first (boolean): Return starred children first (default: 0)
        descending (boolean): Reverse order

    Returns:
        List of members
    """

    parser = reqparse.RequestParser()
    parser.add_argument("nodes", type=strtobool, default=0)
    parser.add_argument("objects", type=strtobool, default=0)
    parser.add_argument("arrange_by", default="")
    parser.add_argument("page", type=int, default=0)
    parser.add_argument("request_id", default=None)
    parser.add_argument("starred_first", type=strtobool, default=1)
    parser.add_argument("descending", type=strtobool, default=0)
    arguments = parser.parse_args(strict=True)

    return _get_node_members(node_id=node_id, **arguments)


@api.route("/nodes/<int:node_id>/progress", methods=["GET"])
def get_node_stats(node_id):
    """
    Return progress information about this node.

    URL parameters:
        node_id (int): ID of a node

    Request parameters:
        log (str): Save an entry to the log?

    Returns:
        JSON-dict
    """

    parser = reqparse.RequestParser()
    parser.add_argument("log", default=None)
    arguments = parser.parse_args(strict=True)

    with database.engine.connect() as connection:
        tree = Tree(connection)

        with connection.begin():
            progress = tree.calculate_progress(node_id)

            if arguments["log"] is not None:
                log(
                    connection,
                    "progress-{}".format(arguments["log"]),
                    node_id=node_id,
                    data=json_dumps(progress),
                )

            return jsonify(progress)


@api.route("/nodes/<int:node_id>/members", methods=["POST"])
def post_node_members(node_id):
    data = request.get_json()

    object_ids = [d["object_id"] for d in data if "object_id" in d]
    node_ids = [d["node_id"] for d in data if "node_id" in d]

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

        flags = {k: request.args.get(k, 0, strtobool) for k in ("include_children",)}

        node = tree.get_node(node_id)

        log(connection, "get_node", node_id=node_id)

        result = _node(tree, node, **flags)

        return jsonify(result)


@api.route("/nodes/<int:node_id>", methods=["PATCH"])
def patch_node(node_id):
    with database.engine.connect() as connection:
        tree = Tree(connection)

        data = request.get_json()
        flags = {k: request.args.get(k, 0, strtobool) for k in ("include_children",)}

        # TODO: Use argparse
        if "starred" in data:
            data["starred"] = strtobool(str(data["starred"]))

        if "parent_id" in data:
            raise ValueError(
                "parent_id must not be set directly, use /nodes/<node_id>/adopt."
            )

        with connection.begin():
            tree.update_node(node_id, data)

            log(
                connection,
                "update_node({})".format(json.dumps(data, sort_keys=True)),
                node_id=node_id,
            )

            node = tree.get_node(node_id, True)

        result = _node(tree, node, **flags)

        return jsonify(result)


@api.route("/nodes/<int:parent_id>/adopt_members", methods=["POST"])
def node_adopt_members(parent_id):
    """
    Adopt a list of nodes.

    URL parameters:
        parent_id (int): ID of the node that accepts new members.

    Request parameters:
        members: List of nodes ({node_id: ...}) and objects ({object_id: ...}).

    Returns:
        Nothing.
    """
    with database.engine.connect() as connection:
        tree = Tree(connection)

        members = request.get_json()["members"]

        node_ids = [int(m["node_id"]) for m in members if "node_id" in m]
        object_ids = [m["object_id"] for m in members if "object_id" in m]

        with connection.begin():
            tree.relocate_nodes(node_ids, parent_id)
            tree.relocate_objects(object_ids, parent_id)

        print(
            "Node {} adopted {} nodes and {} objects.".format(
                parent_id, len(node_ids), len(object_ids)
            )
        )

        return jsonify({})


@api.route("/nodes/<int:node_id>/accept_recommended_objects", methods=["POST"])
def accept_recommended_objects(node_id):
    """
    Accept recommended objects.

    URL parameters:
        node_id (int): ID of the node that accepts recommendations

    Request parameters:
        request_id: URL of the recommendations.
        rejected_members: Rejected members.
        last_page: Last page of accepted recommendations.
        log_data (optional): Additional data to be stored in the log (only if SAVE_RECOMMENDATION_STATS!)

    Returns:
        Nothing.
    """

    parameters = request.get_json()

    print(parameters)

    with Timer("accept_recommended_objects") as t:

        with t.child("assemble set of rejected objects"):
            rejected_object_ids = set(
                m[1:] for m in parameters["rejected_members"] if m.startswith("o")
            )

        with t.child("assemble list of accepted objects"):
            object_ids = []
            for page in range(parameters["last_page"] + 1):
                response = _node_get_recommended_objects(
                    node_id=node_id, request_id=parameters["request_id"], page=page
                )
                page_object_ids = (
                    v["object_id"] for v in json.loads(response.data.decode())["data"]
                )
                object_ids.extend(page_object_ids)

        # Save list of objects to enable calculation of Average Precision and the like
        if app.config.get("SAVE_RECOMMENDATION_STATS", False):
            print("Saving accept-reject stats...")
            with t.child("Save accept-reject stats") as t2:
                with t2.child("calc rejected"):
                    rejected = [o in rejected_object_ids for o in object_ids]
                with t2.child("assemble DataFrame"):
                    data = pd.DataFrame({"object_id": object_ids, "rejected": rejected})

                data_fn = os.path.join(
                    app.config["PROJECT_EXPORT_DIR"],
                    "{:%Y-%m-%d-%H-%M-%S}--accept-reject--{}.csv".format(
                        datetime.now(), node_id
                    ),
                )
                with t2.child("write data"):
                    data.to_csv(data_fn, index=False)

        with t.child("filter accepted objects"):
            # Filter object_ids
            object_ids = [o for o in object_ids if o not in rejected_object_ids]

        # print(object_ids)

        # Assemble log
        log_data = {
            "n_accepted": len(object_ids),
            "n_rejected": len(rejected_object_ids),
        }

        # Store additional log data
        addlog_data = parameters.get("log_data")
        if isinstance(addlog_data, dict):
            log_data.update(addlog_data)
        elif addlog_data is not None:
            raise ValueError(
                "Parameter log_data should be a dict, got a {}!".format(
                    type(addlog_data)
                )
            )

        with database.engine.connect() as connection:
            tree = Tree(connection)
            with t.child("save accepted/rejected to database"), connection.begin():
                tree.relocate_objects(object_ids, node_id)
                tree.reject_objects(node_id, rejected_object_ids)

            log(
                connection,
                "accept_recommended_objects",
                node_id=node_id,
                data=json_dumps(log_data),
            )

        print(
            "Node {} adopted {} objects and rejected {} objects.".format(
                node_id, len(object_ids), len(rejected_object_ids)
            )
        )

        return jsonify({})


@cache_serialize_page(".node_get_recommended_children", page_size=20)
def _node_get_recommended_children(node_id, max_n):
    with database.engine.connect() as connection:
        tree = Tree(connection)
        result = [_node(tree, c) for c in tree.recommend_children(node_id, max_n=max_n)]
        return result


@api.route("/nodes/<int:node_id>/recommended_children", methods=["GET"])
def node_get_recommended_children(node_id):
    """
    Recommend children for this node.

    URL parameters:
        node_id (int): ID of the node.

    Request parameters (GET):
        page (int): Page number (default 0)
        request_id (str, optional): Identification string for the current request collection.
    """
    parser = reqparse.RequestParser()
    parser.add_argument("page", type=int, default=0)
    parser.add_argument("max_n", type=int, default=100)
    parser.add_argument("request_id", default=None)
    arguments = parser.parse_args(strict=True)

    # Limit max_n
    arguments.max_n = max(arguments.max_n, 1000)

    return _node_get_recommended_children(node_id=node_id, **arguments)


@cache_serialize_page(".node_get_recommended_objects", page_size=50)
def _node_get_recommended_objects(node_id=None, max_n=None):
    with database.engine.connect() as connection:
        tree = Tree(connection)

        result = [_object(o) for o in tree.recommend_objects(node_id, max_n)]

        return result


@api.route("/nodes/<int:node_id>/recommended_objects", methods=["GET"])
def node_get_recommended_objects(node_id):
    """
    Recommend objects for this node.

    URL parameters:
        node_id (int): ID of the node.

    Request parameters (GET):
        page (int): Page number (default 0)
        request_id (str, optional): Identification string for the current request collection.
        max_n (int): Maximum number of recommended objects.
    """
    parser = reqparse.RequestParser()
    parser.add_argument("page", type=int, default=0)
    parser.add_argument("max_n", type=int, default=100)
    parser.add_argument("request_id", default=None)
    arguments = parser.parse_args(strict=True)

    # Limit max_n
    arguments.max_n = max(arguments.max_n, 1000)

    return _node_get_recommended_objects(node_id=node_id, **arguments)


@api.route("/nodes/<int:node_id>/tip", methods=["GET"])
def node_get_tip(node_id):
    with database.engine.connect() as connection:
        tree = Tree(connection)

        return jsonify(tree.get_tip(node_id))


@api.route("/nodes/<int:node_id>/next", methods=["GET"])
def node_get_next(node_id):
    parser = reqparse.RequestParser()
    parser.add_argument("leaf", type=strtobool, default=False)
    arguments = parser.parse_args(strict=True)

    print(arguments)

    with database.engine.connect() as connection:
        tree = Tree(connection)

        # Descend if the successor is not approved
        # Rationale: Approval is for a whole subtree.
        def recurse(_, s):
            return s.c.approved == False

        # Filter descendants that are not approved
        def filter(subtree):
            return subtree.c.approved == False

        return jsonify(
            tree.get_next_node(
                node_id, leaf=arguments["leaf"], recurse_cb=recurse, filter=filter
            )
        )


@api.route("/nodes/<int:node_id>/next_unfilled", methods=["GET"])
def node_get_next_unfilled(node_id):
    parser = reqparse.RequestParser()
    parser.add_argument("leaf", type=strtobool, default=False)
    parser.add_argument("preferred_first", type=strtobool, default=False)
    parser.add_argument("order_by", type=str, default=None)
    arguments = parser.parse_args(strict=True)

    print(arguments)

    order_by = arguments["order_by"]

    if order_by is None:
        # Get default value from config
        order_by = api.config["NODE_GET_NEXT_UNFILLED_ORDER_BY"]

    with database.engine.connect() as connection:
        tree = Tree(connection)

        # Consolidate whole tree to populate cached values
        node = tree.get_node(node_id)
        root_id = tree.get_root_id(node["project_id"])
        tree.consolidate_node(root_id, depth="full")

        # Filter descendants that are approved and unfilled
        def filter(subtree):
            return (
                (subtree.c.approved == True)
                & (subtree.c.filled == False)
                & (subtree.c._prototypes != None)
            )

        return jsonify(
            tree.get_next_node(
                node_id,
                leaf=arguments["leaf"],
                preferred_first=arguments["preferred_first"],
                order_by=order_by,
                filter=filter,
            )
        )


@api.route("/nodes/<int:node_id>/n_sorted", methods=["GET"])
def node_get_n_sorted(node_id):
    with database.engine.connect() as connection:
        tree = Tree(connection)

        nodes = tree.get_minlevel_starred(node_id)

        n_sorted = sum(n["_n_objects_deep"] for n in nodes)

        return jsonify(n_sorted)


@api.route("/nodes/<int:node_id>/merge_into", methods=["POST"])
def post_node_merge_into(node_id):
    """
    Merge a node into another node.

    URL parameters:
        node_id: Node that is merged.

    Request parameters (body):
        dest_node_id: Node that absorbs the children and objects.
    """
    with database.engine.connect() as connection:
        tree = Tree(connection)

        data = request.get_json()

        print(data)

        # TODO: Unapprove
        tree.merge_node_into(node_id, data["dest_node_id"])

        log(
            connection,
            "merge_node_into({}, {})".format(node_id, data["dest_node_id"]),
            node_id=data["dest_node_id"],
        )

        return jsonify(None)


@api.route("/nodes/<int:node_id>/classify", methods=["POST"])
def post_node_classify(node_id):
    """
    Classify the members of a node into their starred siblings.

    URL parameters:
        node_id: Parent of the classified members.

    GET parameters:
        nodes (boolean): Classify nodes? (Default: False)
        objects (boolean): Classify objects? (Default: False)
        safe (boolean): Perform safe classification (Default: False)
        subnode (boolean): Move classified objects into a child of the target node. (Default: False)
    """

    flags = {
        k: request.args.get(k, 0, strtobool)
        for k in ("nodes", "objects", "safe", "subnode")
    }

    print(flags)

    n_predicted_children = 0
    n_predicted_objects = 0

    with database.engine.connect() as connection:
        tree = Tree(connection)

        # Split children into starred and unstarred
        with connection.begin():
            children = tree.get_children(node_id)

            starred = []
            unstarred = []
            for c in children:
                (starred if c["starred"] else unstarred).append(c)

            starred_centroids = np.array([c["_centroid"] for c in starred])

            print("|starred_centroids|", np.linalg.norm(starred_centroids, axis=1))

            # Initialize classifier
            classifier = Classifier(starred_centroids)

            if flags["subnode"]:

                def _subnode_for(node_id):
                    return tree.create_node(parent_id=node_id, name="classified")

                target_nodes = keydefaultdict(_subnode_for)
            else:
                target_nodes = keydefaultdict(lambda k: k)

            if flags["nodes"]:
                unstarred_centroids = np.array([c["_centroid"] for c in unstarred])
                unstarred_ids = np.array([c["node_id"] for c in unstarred])

                # Predict unstarred children (if any)
                n_unstarred = len(unstarred_centroids)
                if n_unstarred > 0:
                    print(
                        "Predicting {} unstarred children of {}...".format(
                            n_unstarred, node_id
                        )
                    )
                    type_predicted = classifier.classify(
                        unstarred_centroids, safe=flags["safe"]
                    )

                    for i, starred_node in enumerate(starred):
                        nodes_to_move = [
                            int(n) for n in unstarred_ids[type_predicted == i]
                        ]

                        if len(nodes_to_move):
                            target_node_id = target_nodes[starred_node["node_id"]]
                            tree.relocate_nodes(
                                nodes_to_move, target_node_id, unapprove=True
                            )

                    n_predicted_children = np.sum(type_predicted > -1)

            if flags["objects"]:
                # Predict objects
                objects = tree.get_objects(node_id)
                print("Predicting {} objects of {}...".format(len(objects), node_id))
                object_vectors = np.array([o["vector"] for o in objects])
                object_ids = np.array([o["object_id"] for o in objects])

                type_predicted = classifier.classify(object_vectors, safe=flags["safe"])

                for i, starred_node in enumerate(starred):
                    objects_to_move = [str(o) for o in object_ids[type_predicted == i]]
                    if len(objects_to_move):
                        target_node_id = target_nodes[starred_node["node_id"]]
                        print(
                            "Moving objects {!r} -> {}".format(
                                objects_to_move, target_node_id
                            )
                        )
                        tree.relocate_objects(
                            objects_to_move, target_node_id, unapprove=True
                        )

                n_predicted_objects = np.sum(type_predicted > -1)

            log(
                connection,
                "classify_members(nodes={nodes},objects={objects})".format(**flags),
                node_id=node_id,
            )

            return jsonify(
                {
                    "n_predicted_children": int(n_predicted_children),
                    "n_predicted_objects": int(n_predicted_objects),
                }
            )


@api.route("/log", methods=["POST"])
def create_log_entry():
    log_data = LogSchema().load(request.get_json())

    print("Log:", log_data)

    with database.engine.connect() as connection:
        log(
            connection,
            log_data["action"],
            node_id=log_data["node_id"],
            reverse_action=log_data["reverse_action"],
            data=json_dumps(log_data["data"]),
        )

    return jsonify({})


@api.route("/jobs", methods=["POST"])
def create_job():
    data = JobSchema().load(request.get_json())

    fun = getattr(background, data["name"])

    if not background.validate_background_job(fun):
        raise werkzeug.exceptions.NotImplemented()

    job = fun.queue(**data["kwargs"])

    print(job)

    job_url = url_for(".get_job", job_id=job.id)

    data["job"] = job

    return Response(JobSchema().dumps(data), status=202, headers={"Location": job_url})


@api.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    job = rq.get_queue().fetch_job(job_id)

    if job is None:
        raise werkzeug.exceptions.NotFound()

    # job = database.session.query(models.Job).get(job_id)

    data = {"job": job}

    result = JobSchema().dump(data)

    return jsonify(result)
