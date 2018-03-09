from setuptools import setup

setup(
    name='cluster_labeling',
    packages=['cluster_labeling'],
    include_package_data=True,
    install_requires=[
        'flask',
        'flask_restful'
    ],
)
