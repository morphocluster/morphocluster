import os

# Redis (LRU for caching)
REDIS_LRU_URL = "redis://redis-lru:6379/0"

# Redis for rq
RQ_REDIS_URL = "redis://redis-rq:6379/0"

# Database connection
SQLALCHEMY_DATABASE_URI = (
    "postgresql://morphocluster:morphocluster@postgres/morphocluster"
)

PROJECT_EXPORT_DIR = os.environ.get("PROJECT_EXPORT_DIR", "/data/export")

# ORDER BY clause for node_get_next_unfilled
NODE_GET_NEXT_UNFILLED_ORDER_BY = "largest"

PREFERRED_URL_SCHEME = os.environ.get("PREFERRED_URL_SCHEME", None)
