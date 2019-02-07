"""Introduce nodes._prototypes

Revision ID: 8de87a72da91
Revises: 4678451894c9
Create Date: 2019-02-06 17:08:28.723229

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8de87a72da91'
down_revision = '4678451894c9'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('nodes', sa.Column(
        '_prototypes', sa.PickleType(), nullable=True))


def downgrade():
    op.drop_column('nodes', '_prototypes')
