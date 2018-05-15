'''
Created on 27.03.2018

@author: mschroeder
'''

import redis

redis_store = redis.StrictRedis(host='localhost', port=6379, db=0)