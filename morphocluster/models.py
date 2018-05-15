'''
Created on 13.03.2018

@author: mschroeder
'''
from sqlalchemy import Table, Column, ForeignKey, Index

from morphocluster.database import metadata
import datetime

from sqlalchemy.types import Integer, BigInteger, String, DateTime, PickleType, Boolean, Text
from sqlalchemy.sql.schema import UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import func

#: :type objects: sqlalchemy.sql.schema.Table
objects = Table('objects', metadata,
    Column('object_id', String, primary_key = True),
    Column('path', String, nullable = False),
    Column('vector', PickleType, nullable = True)
)

#: :type projects: sqlalchemy.sql.schema.Table
projects = Table('projects', metadata,
    Column('project_id', Integer, primary_key = True),
    Column('name', String),
    Column('creation_date', DateTime, default = datetime.datetime.now)
)

#: :type nodes: sqlalchemy.sql.schema.Table
nodes = Table('nodes', metadata,
    Column('node_id', BigInteger, primary_key = True),
    Column('orig_id', BigInteger, nullable = True),
    Column('project_id', None, ForeignKey('projects.project_id'), nullable = False),
    Column('parent_id', None, ForeignKey('nodes.node_id'), nullable = True),
    Column('name', String),
    Column('starred', Boolean, default = False, nullable = False),
    
    #===========================================================================
    # The following fields are cached values
    #===========================================================================
    # Pickle-serialized numpy array
    Column('_centroid', PickleType, nullable = True),
    # object_ids of type objects representative for all descendants (used as preview)
    Column('_type_objects', ARRAY(String), nullable = True),
    # object_ids of type objects directly under this node
    Column('_own_type_objects', ARRAY(String), nullable = True),
    # Number of objects below this node
    Column('_recursive_n_objects', BigInteger, nullable = True),
    
    # An orig_id must be unique inside a project
    Index('idx_orig_proj', 'orig_id', 'project_id', unique = True),
    
    # A node may not be its own child
    CheckConstraint("node_id != parent_id")
)

nodes_objects = Table('nodes_objects', metadata,
    Column('node_id', None, ForeignKey('nodes.node_id', ondelete="CASCADE"), nullable = False),
    Column('project_id', None, ForeignKey('projects.project_id', ondelete="CASCADE"), nullable = False),
    Column('object_id', None, ForeignKey('objects.object_id', ondelete="CASCADE"), nullable = False),
    UniqueConstraint('project_id', 'object_id')
)

users = Table('users', metadata,
    Column('username', String, primary_key = True),
    Column('pwhash', String)
)

log = Table("log", metadata,
    Column('log_id', BigInteger, primary_key = True),
    Column("timestamp", DateTime(timezone=True), server_default=func.now()),
    Column('node_id', None, ForeignKey('nodes.node_id', ondelete="SET NULL"), nullable = True),
    Column('username', None, ForeignKey('users.username', ondelete="SET NULL"), nullable = True),
    Column('action', Text, nullable = False),
    Column('reverse_action', Text, nullable = True),
)

#===============================================================================
# categories = Table('names', metadata,
#     Column('name', String, primary_key = True),
#     Column('ecotaxa_id', String)
# )
#===============================================================================
