from setuptools import setup

setup(
    name='morphocluster',
    packages=['morphocluster'],
    include_package_data=True,
    install_requires=[
        'flask',
        'psycopg2-binary',
        'pandas',
        'sqlalchemy',
        'etaprogress',
        'h5py',
        'scikit-learn',
        'scipy',
        'redis',
        'hiredis',
        'flask-restful',
        'alembic'
    ],
)
