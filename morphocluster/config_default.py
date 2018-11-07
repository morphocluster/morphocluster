# Redis
REDIS_URL = "redis://:@localhost:6379/0"

# Database
SQLALCHEMY_DATABASE_URI = "postgresql://morphocluster:morphocluster@localhost/morphocluster"
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_DATABASE_OPTIONS = {
    'connect_args': {
        "options": "-c statement_timeout=240s"
    }
}

# Project export directory
PROJECT_EXPORT_DIR = "/tmp"
