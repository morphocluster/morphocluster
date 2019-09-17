#!/bin/sh

. /opt/conda/etc/profile.d/conda.sh
conda activate morphocluster

export FLASK_APP=morphocluster
export MORPHOCLUSTER_SETTINGS=config_docker.py

echo Waiting for Postgres...
./wait-for postgres:5432
# echo Waiting for Redis
# ./wait-for redis:6379

flask db upgrade

/usr/bin/supervisord --nodaemon -c /etc/supervisor/supervisord.conf