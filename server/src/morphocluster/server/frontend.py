"""
Frontend blueprint
"""

import datetime
import json

from flask import current_app
from flask.blueprints import Blueprint

frontend = Blueprint(
    "frontend", __name__, static_folder="frontend/dist", static_url_path=""
)

@frontend.route("/config.js")
def get_config_js():
    content = "window.config = " + json.dumps({k: v for k,v in current_app.config.items() if k.startswith("FRONTEND_")}) + ";"
    return current_app.response_class(content, mimetype='text/javascript')


@frontend.route("/")
@frontend.route("/p")
@frontend.route("/p/<path:path>")
def index(path=None):
    response = frontend.send_static_file("index.html")
    del response.headers["Expires"]
    del response.headers["ETag"]
    response.headers["Last-Modified"] = datetime.datetime.now()
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    print(response.headers)
    return response
