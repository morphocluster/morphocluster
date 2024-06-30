"""Store vectors as CUBE points

Revision ID: 762c3a983d96
Revises: fe6fec6b70a6
Create Date: 2021-10-28 08:59:45.382312

"""
from alembic import op
import sqlalchemy as sa
from morphocluster.server.sql.types import Point
from sqlalchemy.types import LargeBinary

# revision identifiers, used by Alembic.
revision = "762c3a983d96"
down_revision = "fe6fec6b70a6"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column("objects", "vector")
    op.add_column("objects", sa.Column("vector", Point(numpy=True), nullable=True))


def downgrade():
    op.drop_column("objects", "vector")
    sa.Column("vector", LargeBinary, autoincrement=False, nullable=True),
