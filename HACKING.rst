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


Development environment
-----------------------

The repository contains a `.devcontainer` configuration for `VS Code <https://code.visualstudio.com/>`_.
This provides a standardized development environment.

We suggest setting `DOCKER_BUILDKIT=1` and `COMPOSE_DOCKER_CLI_BUILD=1` in your host environment so that you benefit from package caching when rebuilding the docker containers.

To tear down the build environment, do `docker-compose down --project-name morphocluster -v --rmi all`.
This removes all data outside the project tree.

Building the image
------------------

Run `./build_docker.sh` to build the docker image.
Prepend with `MORPHOCLUSTER_CUDA=10.2` for CUDA 10.2 support.

Frontend Development
--------------------

Install a current Node.js and npm using:

    nvm install --lts --latest-npm