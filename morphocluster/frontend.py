from flask.blueprints import Blueprint

frontend = Blueprint("frontend", __name__,
                static_folder="frontend_static", static_url_path="")

@frontend.route("/")
def index():
    return frontend.send_static_file('index.html')