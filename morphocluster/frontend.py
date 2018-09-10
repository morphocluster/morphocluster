from flask.blueprints import Blueprint

frontend = Blueprint("frontend", __name__,
                     static_folder="frontend_static", static_url_path="")


@frontend.route("/")
@frontend.route("/p")
@frontend.route("/p/<path:path>")
def index(path=None):
    return frontend.send_static_file('index.html')
