'''
Created on 23.05.2018

@author: mschroeder
'''

from morphocluster.sqlalchemy_connection import SQLAlchemyConnection
from flask_redis import FlaskRedis
from flask_migrate import Migrate
from flask_rq2 import RQ

database = SQLAlchemyConnection()
redis_lru = FlaskRedis(config_prefix="REDIS_LRU")
migrate = Migrate()
rq = RQ()
