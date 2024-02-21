#!/bin/sh

# . /opt/conda/etc/profile.d/conda.sh
micromamba activate base

export FLASK_APP=morphocluster

echo Waiting for Postgres...
./wait-for postgres:5432
# echo Waiting for Redis
# ./wait-for redis:6379

flask db upgrade

/usr/bin/supervisord --nodaemon -c /etc/supervisor/supervisord.conf