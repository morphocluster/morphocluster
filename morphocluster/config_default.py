from environs import Env

_env = Env()
_env.read_env()

# Redis (LRU for caching)
# REDIS_LRU_URL = ...

# Redis for rq
# RQ_REDIS_URL = ...

# Database
SQLALCHEMY_DATABASE_URI = (
    "postgresql://morphocluster:morphocluster@localhost/morphocluster"
)
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_DATABASE_OPTIONS = {"connect_args": {"options": "-c statement_timeout=240s"}}

# Project export directory
PROJECT_EXPORT_DIR = "/tmp"

# Save the results of accept_recommended_objects
# to enable the calculation of scores like average precision
SAVE_RECOMMENDATION_STATS = False

DATASET_PATH = "/data"

FRONTEND_SHOW_MEMBER_TITLE = _env.bool("FRONTEND_SHOW_MEMBER_TITLE", default=True)