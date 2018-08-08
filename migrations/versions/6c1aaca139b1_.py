"""empty message

Revision ID: 6c1aaca139b1
Revises: 
Create Date: 2018-05-08 14:35:50.224919

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6c1aaca139b1'
down_revision = '0a04bbfe404b'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('nodes', sa.Column('_recursive_n_objects', sa.BigInteger(), nullable=True))


def downgrade():
    op.drop_column('nodes', '_recursive_n_objects')
