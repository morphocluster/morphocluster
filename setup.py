from setuptools import setup

setup(
    name='cluster_labeling',
    packages=['cluster_labeling'],
    include_package_data=True,
    install_requires=[
        'flask',
        'psycopg2-binary',
        'pandas',
        'sqlalchemy',
        'etaprogress',
        'tables',
        'scikit-learn',
        'scipy',
        'redis',
        'hiredis'
    ],
)
