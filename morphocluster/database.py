'''
Created on 13.03.2018

@author: mschroeder
'''

from sqlalchemy import create_engine, MetaData

#engine = create_engine('sqlite:///test.db', convert_unicode=True)
#: :type engine: sqlalchemy.engine.base.Engine     
engine = create_engine('postgresql://clusterlabeling:clusterlabeling@localhost/clusterlabeling',
                       use_batch_mode = True,
                       echo = False)

#: :type metadata: sqlalchemy.sql.schema.MetaData
metadata = MetaData(bind=engine)