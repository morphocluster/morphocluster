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
        'h5py>=2.8.0',
        'scikit-learn',
        'scipy',
        'redis',
        'hiredis',
        'flask-restful',
        'alembic',
        'Flask-SQLAlchemy',
        'flask-redis',
        'Flask-Migrate',
        'timer_cm',
        'fire',
        'marshmallow>=3.0.0'
    ],
)
