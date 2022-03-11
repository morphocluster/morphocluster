#!/bin/sh

cp /authorized_keys /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

. /opt/conda/etc/profile.d/conda.sh
conda activate base

export FLASK_APP=morphocluster

echo Waiting for Postgres...
./wait-for postgres:5432
# echo Waiting for Redis
# ./wait-for redis:6379

flask db upgrade

/usr/bin/supervisord --nodaemon -c /etc/supervisor/supervisord.conf