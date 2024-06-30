"""Initial setup

Revision ID: 0a04bbfe404b
Revises: 52928669f064
Create Date: 2018-08-08 10:32:07.572893

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0a04bbfe404b'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('projects',
        sa.Column('project_id', sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column('name', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('creation_date', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
        sa.PrimaryKeyConstraint('project_id', name='projects_pkey'),
        postgresql_ignore_search_path=False
    )
    
    op.create_table('nodes',
        sa.Column('node_id', sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column('orig_id', sa.BIGINT(), autoincrement=False, nullable=True),
        sa.Column('project_id', sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column('parent_id', sa.BIGINT(), autoincrement=False, nullable=True),
        sa.Column('name', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column('starred', sa.BOOLEAN(), autoincrement=False, nullable=False),
        sa.Column('_centroid', sa.PickleType(), autoincrement=False, nullable=True),
        sa.Column('_type_objects', postgresql.ARRAY(sa.VARCHAR()), autoincrement=False, nullable=True),
        sa.Column('_own_type_objects', postgresql.ARRAY(sa.VARCHAR()), autoincrement=False, nullable=True),
        sa.CheckConstraint('node_id <> parent_id', name='nodes_check'),
        sa.ForeignKeyConstraint(['parent_id'], ['nodes.node_id'], name='nodes_parent_id_fkey'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.project_id'], name='nodes_project_id_fkey'),
        sa.PrimaryKeyConstraint('node_id', name='nodes_pkey'),
        postgresql_ignore_search_path=False
    )
    op.create_index('idx_orig_proj', 'nodes', ['orig_id', 'project_id'], unique=True)
    
    op.create_table('users',
        sa.Column('username', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column('pwhash', sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.PrimaryKeyConstraint('username', name='users_pkey')
    )
    
    op.create_table('objects',
        sa.Column('object_id', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column('path', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column('vector', postgresql.BYTEA(), autoincrement=False, nullable=True),
        sa.PrimaryKeyConstraint('object_id', name='objects_pkey'),
        postgresql_ignore_search_path=False
    )
    
    op.create_table('nodes_objects',
        sa.Column('node_id', sa.BIGINT(), autoincrement=False, nullable=False),
        sa.Column('project_id', sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column('object_id', sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.ForeignKeyConstraint(['node_id'], ['nodes.node_id'], name='nodes_objects_node_id_fkey', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['object_id'], ['objects.object_id'], name='nodes_objects_object_id_fkey', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.project_id'], name='nodes_objects_project_id_fkey', ondelete='CASCADE'),
        sa.UniqueConstraint('project_id', 'object_id', name='nodes_objects_project_id_object_id_key')
    )
    
    op.create_table('log',
        sa.Column('log_id', sa.BIGINT(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column('node_id', sa.BIGINT(), nullable=True),
        sa.Column('username', sa.VARCHAR(), nullable=True),
        sa.Column('action', sa.TEXT(), nullable=False),
        sa.Column('reverse_action', sa.TEXT(), nullable=True),
        sa.ForeignKeyConstraint(['node_id'], ['nodes.node_id'], name='log_node_id_fkey', ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['username'], ['users.username'], name='log_username_fkey', ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('log_id', name='log_pkey')
    )

def downgrade():
    op.drop_table('nodes_objects')
    op.drop_table('objects')
    op.drop_table('projects')
    op.drop_table('users')
    op.drop_index('idx_orig_proj', table_name='nodes')
    op.drop_table('nodes')
    op.drop_table('log')

