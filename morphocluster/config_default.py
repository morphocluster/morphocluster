from environs import Env

_env = Env()
_env.read_env()

# Redis (LRU for caching)
REDIS_LRU_URL = "redis://redis-lru:6379/0"

# Redis for rq
RQ_REDIS_URL = "redis://redis-rq:6379/0"

# Database connection
SQLALCHEMY_DATABASE_URI = _env.str("MORPHOCLUSTER_DATABASE_URI", default=
    "postgresql://morphocluster:morphocluster@postgres/morphocluster"
)
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_DATABASE_OPTIONS = {"connect_args": {"options": "-c statement_timeout=240s"}}

# Project export directory
PROJECT_EXPORT_DIR = "/data/export"

# Save the results of accept_recommended_objects
# to enable the calculation of scores like average precision
SAVE_RECOMMENDATION_STATS = False

DATASET_PATH = _env.str("DATASET_PATH", default="/data")

# ORDER BY clause for node_get_next_unfilled
NODE_GET_NEXT_UNFILLED_ORDER_BY = "largest"

## Flask configuration
# https://flask.palletsprojects.com/en/2.2.x/config/#PREFERRED_URL_SCHEME
PREFERRED_URL_SCHEME = _env.str("PREFERRED_URL_SCHEME", default=None)

# https://flask.palletsprojects.com/en/2.2.x/config/#TRAP_BAD_REQUEST_ERRORS
TRAP_BAD_REQUEST_ERRORS = _env.bool("TRAP_BAD_REQUEST_ERRORS", default=False)

## Frontend configuration
## Accessible as window.config.<key>
# Show the title (object_id, node_id) of cluster members
FRONTEND_SHOW_MEMBER_TITLE = _env.bool("FRONTEND_SHOW_MEMBER_TITLE", default=True)
