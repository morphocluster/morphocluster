'''
Created on 13.03.2018

@author: mschroeder
'''
from sqlalchemy import Table, Column, ForeignKey, Index

import datetime

from sqlalchemy.types import Integer, BigInteger, String, DateTime, PickleType, Boolean, Text, Float
from sqlalchemy.sql.schema import UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import func
from morphocluster.extensions import database

metadata = database.metadata

#: :type objects: sqlalchemy.sql.schema.Table
objects = Table('objects', metadata,
    Column('object_id', String, primary_key=True),
    Column('path', String, nullable=False),
    Column('vector', PickleType, nullable=True),
    Column('rand', Float, server_default=func.random())
)

#: :type projects: sqlalchemy.sql.schema.Table
projects = Table('projects', metadata,
    Column('project_id', Integer, primary_key=True),
    Column('name', String),
    Column('creation_date', DateTime,
        default=datetime.datetime.now),
    Column('visible', Boolean, nullable=False, server_default="t"),
)

#: :type nodes: sqlalchemy.sql.schema.Table
nodes = Table('nodes', metadata,
    Column('node_id', BigInteger, primary_key=True),
    Column('orig_id', BigInteger, nullable=True),
    Column('project_id', None,
            ForeignKey('projects.project_id', ondelete="CASCADE"),
            index=True, nullable=False),
    Column('parent_id', None,
            ForeignKey('nodes.node_id', ondelete="SET NULL"),
            index=True, nullable=True),
    Column('name', String),
    Column('starred', Boolean, default=False, nullable=False),
    Column('approved', Boolean, default=False,
            nullable=False, server_default="f"),

    # ===========================================================================
    # Super Node support
    # ===========================================================================
    Column('superparent_id', None,
            ForeignKey('nodes.node_id', ondelete="SET NULL"),
            index=True, nullable=True),

    # ===========================================================================
    # The following fields are cached values
    # ===========================================================================
    # Mean feature vector
    Column('_centroid', PickleType, nullable=True),
    # object_ids of type objects representative for all descendants (used as preview)
    Column('_type_objects', ARRAY(String), nullable=True),
    # object_ids of type objects directly under this node (used as preview for the node's objects)
    Column('_own_type_objects', ARRAY(String), nullable=True),
    # Number of children of this node
    Column('_n_children', BigInteger, nullable=True),
    # Number of objects directly below this node
    Column('_n_objects', BigInteger, nullable=True),
    # Number of all objects anywhere below this node
    Column('_n_objects_deep', BigInteger, nullable=True),
    # Covariance
    Column('_covariance', PickleType, nullable=True),

    # Validity of cached values
    Column('cache_valid', Boolean,
            nullable=False, server_default="f"),

    # An orig_id must be unique inside a project
    Index('idx_orig_proj', 'orig_id', 'project_id', unique=True),

    # A node may not be its own child
    CheckConstraint("node_id != parent_id")
)

nodes_objects = Table('nodes_objects', metadata,
    Column('node_id', None,
            ForeignKey('nodes.node_id', ondelete="CASCADE"),
            index=True, nullable=False),
    Column('project_id', None,
            ForeignKey('projects.project_id',
                    ondelete="CASCADE"),
            index=True, nullable=False),
    Column('object_id', None,
            ForeignKey('objects.object_id',
                    ondelete="CASCADE"),
            nullable=False),
    UniqueConstraint('project_id', 'object_id')
)

users = Table('users', metadata,
    Column('username', String, primary_key=True),
    Column('pwhash', String)
)

log = Table("log", metadata,
    Column('log_id', BigInteger, primary_key=True),
    Column("timestamp", DateTime(timezone=True),
            server_default=func.now()),
    Column('node_id', None, ForeignKey(
        'nodes.node_id', ondelete="SET NULL"), nullable=True),
    Column('username', None, ForeignKey(
        'users.username', ondelete="SET NULL"), nullable=True),
    Column('action', Text, nullable=False),
    Column('reverse_action', Text, nullable=True),
)

# ===============================================================================
# categories = Table('names', metadata,
#     Column('name', String, primary_key = True),
#     Column('ecotaxa_id', String)
# )
# ===============================================================================
