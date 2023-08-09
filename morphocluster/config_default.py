import posixpath

from environs import Env

_env = Env()
_env.read_env()

# Redis (LRU for caching)
REDIS_LRU_URL = "redis://redis-lru:6379/0"

# Redis for rq
RQ_REDIS_URL = "redis://localhost:6379/0"

# Database
SQLALCHEMY_DATABASE_URI = (
    "postgresql://morphocluster:morphocluster@localhost/morphocluster"
)
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_DATABASE_OPTIONS = {"connect_args": {"options": "-c statement_timeout=240s"}}

# Project export directory
PROJECT_EXPORT_DIR = "/tmp"

RECLUSTER_FEATURES = [
    "/data1/mschroeder/NoveltyDetection/Results/CrossVal/2018-02-06-12-39-56/split-2/collection_train_2_val.h5",
    "/data1/mschroeder/NoveltyDetection/Results/CrossVal/2018-02-06-12-39-56/split-2/collection_unlabeled_1M.h5",
]

# Save the results of accept_recommended_objects
# to enable the calculation of scores like average precision
SAVE_RECOMMENDATION_STATS = False

# Directory where the data will be stored (absolute or relative to app.root_path)
DATA_DIR = ""
