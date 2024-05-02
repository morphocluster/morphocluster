"""
Created on 13.03.2018

@author: mschroeder
"""
import datetime

# pylint: disable=W,C,R
from sqlalchemy import Column, ForeignKey, Index, Table
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import func
from sqlalchemy.sql.schema import CheckConstraint, UniqueConstraint
from sqlalchemy.types import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Integer,
    PickleType,
    String,
    Text,
)

from morphocluster.extensions import database as db
from morphocluster.sql.types import Point

metadata = db.metadata

#: :type objects: sqlalchemy.sql.schema.Table
objects = Table(
    "objects",
    metadata,
    Column("object_id", String, primary_key=True),
    Column("path", String, nullable=False),
    Column("vector", Point(numpy=True), nullable=True),
    Column("rand", Float, server_default=func.random()),
)

#: :type projects: sqlalchemy.sql.schema.Table
projects = Table(
    "projects",
    metadata,
    Column("project_id", Integer, primary_key=True),
    Column("name", String),
    Column("creation_date", DateTime, default=datetime.datetime.now),
    Column("visible", Boolean, nullable=False, server_default="t"),
)

#: :type nodes: sqlalchemy.sql.schema.Table
nodes = Table(
    "nodes",
    metadata,
    Column("node_id", BigInteger, primary_key=True),
    Column("orig_id", BigInteger, nullable=True),
    Column(
        "project_id",
        None,
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    ),
    Column(
        "parent_id",
        None,
        ForeignKey("nodes.node_id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    ),
    Column("name", String),
    Column("starred", Boolean, default=False, nullable=False, server_default="f"),
    Column("approved", Boolean, default=False, nullable=False, server_default="f"),
    Column("filled", Boolean, default=False, nullable=False, server_default="f"),
    Column("preferred", Boolean, default=False, nullable=False, server_default="f"),
    # ===========================================================================
    # Super Node support
    # ===========================================================================
    Column(
        "superparent_id",
        None,
        ForeignKey("nodes.node_id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    ),
    # ===========================================================================
    # The following fields are cached values
    # ===========================================================================
    # Centroid (single)
    Column("_centroid", PickleType, nullable=True),
    # Prototypes (multiple centroid)
    Column("_prototypes", PickleType, nullable=True),
    # object_ids of type objects representative for all descendants (used as preview)
    Column("_type_objects", ARRAY(String), nullable=True),
    # object_ids of type objects directly under this node (used as preview for the node's objects)
    Column("_own_type_objects", ARRAY(String), nullable=True),
    # Number of children of this node
    Column("_n_children", BigInteger, nullable=True),
    # Number of objects directly below this node
    Column("_n_objects", BigInteger, nullable=True),
    # Number of all objects anywhere below this node
    Column("_n_objects_deep", BigInteger, nullable=True),
    # Validity of cached values
    Column("cache_valid", Boolean, nullable=False, server_default="f"),
    # An orig_id must be unique inside a project
    Index("idx_orig_proj", "orig_id", "project_id", unique=True),
    # A node may not be its own child
    CheckConstraint("node_id != parent_id"),
)

nodes_objects = Table(
    "nodes_objects",
    metadata,
    Column(
        "node_id",
        None,
        ForeignKey("nodes.node_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    ),
    Column(
        "project_id",
        None,
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    ),
    Column(
        "object_id",
        None,
        ForeignKey("objects.object_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    ),
    UniqueConstraint("project_id", "object_id"),
)

nodes_rejected_objects = Table(
    "nodes_rejected_objects",
    metadata,
    Column(
        "node_id",
        None,
        ForeignKey("nodes.node_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    ),
    Column(
        "object_id",
        None,
        ForeignKey("objects.object_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    ),
)

users = Table(
    "users",
    metadata,
    Column("username", String, primary_key=True),
    Column("pwhash", String),
)

log = Table(
    "log",
    metadata,
    Column("log_id", BigInteger, primary_key=True),
    Column("timestamp", DateTime(timezone=True), server_default=func.now()),
    Column(
        "node_id", None, ForeignKey("nodes.node_id", ondelete="SET NULL"), nullable=True
    ),
    Column(
        "username",
        None,
        ForeignKey("users.username", ondelete="SET NULL"),
        nullable=True,
    ),
    Column("action", Text, nullable=False),
    Column("reverse_action", Text, nullable=True),
    Column("data", Text, nullable=True),
)

# ===============================================================================
# categories = Table('names', metadata,
#     Column('name', String, primary_key = True),
#     Column('ecotaxa_id', String)
# )
# ===============================================================================

import enum


class JobState(enum.Enum):
    pass


class Job(db.Model):
    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(String)
    username = Column(None, ForeignKey("users.username"))
