# pylint: disable=W,C,R

import datetime
import enum

from sqlalchemy import Column, ForeignKey, Index, Table
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import func
from sqlalchemy.sql.schema import (
    CheckConstraint,
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
    UniqueConstraint,
)
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
from morphocluster.cube import Cube

metadata = db.metadata

datasets = Table(
    "datasets",
    metadata,
    Column("dataset_id", Integer, primary_key=True),
    Column("name", String),
    Column(
        "owner",
        None,
        ForeignKey("users.username", ondelete="CASCADE", name="users_username_fkey"),
        nullable=False,
    ),
)

#: :type objects: sqlalchemy.sql.schema.Table
objects = Table(
    "objects",
    metadata,
    Column("object_id", String, nullable=False),
    Column("path", String, nullable=False),  # TODO: image_fn
    Column("vector", PickleType, nullable=True),
    # rand is for quasi-random samples (e.g. type object calculation)
    Column("rand", Float, server_default=func.random(), nullable=False),
    Column(
        "dataset_id",
        None,
        ForeignKey(
            "datasets.dataset_id", ondelete="CASCADE", name="datasets_dataset_id_fkey"
        ),
        nullable=False,
    ),
    PrimaryKeyConstraint("dataset_id", "object_id", name="objects_pk"),
    postgresql_partition_by="LIST(dataset_id)",
)

#: :type projects: sqlalchemy.sql.schema.Table
projects = Table(
    "projects",
    metadata,
    Column("project_id", Integer, primary_key=True, nullable=False),
    Column("name", String),
    Column("creation_date", DateTime, default=datetime.datetime.now, nullable=False),
    Column("visible", Boolean, nullable=False, server_default="t"),
    Column(
        "dataset_id",
        None,
        ForeignKey(
            "datasets.dataset_id", ondelete="CASCADE", name="projects_dataset_id_fkey"
        ),
        nullable=False,
    ),
)

#: :type nodes: sqlalchemy.sql.schema.Table
nodes = Table(
    "nodes",
    metadata,
    ## Regular properties
    Column("node_id", BigInteger, nullable=False),
    Column(
        "project_id",
        None,
        ForeignKey(
            "projects.project_id", ondelete="CASCADE", name="projects_project_id_fkey"
        ),
        nullable=False,
    ),
    Column("parent_id", None, index=True, nullable=True),
    Column("name", String),
    Column("starred", Boolean, default=False, nullable=False, server_default="f"),
    Column("approved", Boolean, default=False, nullable=False, server_default="f"),
    Column("filled", Boolean, default=False, nullable=False, server_default="f"),
    Column("preferred", Boolean, default=False, nullable=False, server_default="f"),
    ## Cached values
    # Number of direct children
    Column("n_children_", BigInteger, nullable=True),
    # Recursive number of below nodes
    Column("n_nodes_", BigInteger, nullable=True),
    # Number of objects directly below this node
    Column("n_objects_own_", BigInteger, nullable=True),
    # Number of all objects anywhere below this node
    Column("n_objects_", BigInteger, nullable=True),
    # Recursive sum of vectors
    Column("vector_sum_", PickleType, nullable=True),
    # Sum of own vectors
    Column("vector_sum_own_", PickleType, nullable=True),
    # Recursive mean of vectors
    Column("vector_mean_", Cube(as_numpy=True), nullable=True),
    # Mean of own vectors
    Column("vector_mean_own_", Cube(as_numpy=True), nullable=True),
    # object_ids of type objects directly under this node (used as preview for the node's objects)
    Column("type_objects_own_", ARRAY(String), nullable=True),
    # object_ids of type objects representative for all descendants (used as preview)
    Column("type_objects_", ARRAY(String), nullable=True),
    # Recursive number of (approved|filled|preferred) (objects|nodes)
    Column("n_approved_objects_", BigInteger, nullable=True),
    Column("n_approved_nodes_", BigInteger, nullable=True),
    Column("n_filled_objects_", BigInteger, nullable=True),
    Column("n_filled_nodes_", BigInteger, nullable=True),
    Column("n_preferred_objects_", BigInteger, nullable=True),
    Column("n_preferred_nodes_", BigInteger, nullable=True),
    ## Validity of cached values
    Column("cache_valid", Boolean, nullable=False, server_default="f"),
    ## Contraints
    # Primary key consists of project_id and node_id (in this order)
    # This means that no individual index for project_id alone is needed.
    PrimaryKeyConstraint("project_id", "node_id", name="nodes_pk"),
    # Foreign key for parent
    ForeignKeyConstraint(
        ["project_id", "parent_id"],
        ["nodes.project_id", "nodes.node_id"],
        ondelete="RESTRICT",
        name="nodes_project_id_node_id_fkey",
    ),
)

nodes_objects = Table(
    "nodes_objects",
    metadata,
    Column("project_id", None, nullable=False),
    Column("node_id", None, index=True, nullable=False),
    Column("dataset_id", None, nullable=False),
    Column("object_id", None, nullable=False),
    UniqueConstraint("project_id", "object_id"),
    Index("idx_nodes_objects_project_id_node_id", "project_id", "node_id"),
    ForeignKeyConstraint(
        ["project_id", "node_id"],
        ["nodes.project_id", "nodes.node_id"],
        ondelete="CASCADE",
        name="nodes_project_id_node_id_fkey",
    ),
    ForeignKeyConstraint(
        ["dataset_id", "object_id"],
        ["objects.dataset_id", "objects.object_id"],
        ondelete="CASCADE",
        name="objects_dataset_id_object_id_fkey",
    ),
    postgresql_partition_by="LIST(project_id)",
)

nodes_rejected_objects = Table(
    "nodes_rejected_objects",
    metadata,
    Column("project_id", None, nullable=False),
    Column("node_id", None, index=True, nullable=False),
    Column("dataset_id", None, nullable=False),
    Column("object_id", None, nullable=False),
    ForeignKeyConstraint(
        ["project_id", "node_id"],
        ["nodes.project_id", "nodes.node_id"],
        ondelete="CASCADE",
        name="node_project_id_node_id_fkey",
    ),
    ForeignKeyConstraint(
        ["dataset_id", "object_id"],
        ["objects.dataset_id", "objects.object_id"],
        ondelete="CASCADE",
        name="objects_dataset_id_object_id_fkey",
    ),
    postgresql_partition_by="LIST(project_id)",
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
    Column("node_id", BigInteger),
    # When the project is deleted the log is obsolete
    Column(
        "project_id",
        None,
        ForeignKey(
            "projects.project_id", ondelete="CASCADE", name="projects_project_id_fkey"
        ),
    ),
    Column("username", String),
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


class JobState(enum.Enum):
    pass


class Job(db.Model):
    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(String)
    username = Column(
        None, ForeignKey("users.username", name="users_username_fkey"), nullable=False
    )
