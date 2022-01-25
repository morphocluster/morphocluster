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