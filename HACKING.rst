MorphoCluster Contribution Guide
================================

Additional Development Packages
-------------------------------

    # flask-sqlalchemy-stubs to make the checker happy
    pip install --no-deps git+https://github.com/ssfdust/flask-sqlalchemy-stubs.git

Creating the test environment
-----------------------------

    conda env create -f environment.default.yml -p .venv
    conda activate ./.venv


Branches
--------

The current stable branch is ``0.2.x``.

Development happens on ``master``.

``maintenance/0.1.x`` is for older setups.
