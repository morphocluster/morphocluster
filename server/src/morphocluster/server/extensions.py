"""
Created on 23.05.2018

@author: mschroeder
"""

from flask_sqlalchemy import SQLAlchemy
from flask_redis import FlaskRedis
from flask_migrate import Migrate
from flask_rq2 import RQ

from sqlalchemy.pool import StaticPool

# StaticPool: Use one connection throughout
database = SQLAlchemy(engine_options=dict(poolclass=StaticPool))
redis_lru = FlaskRedis(config_prefix="REDIS_LRU")
migrate = Migrate()
rq = RQ()
