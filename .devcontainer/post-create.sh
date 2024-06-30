#!/usr/bin/env sh

pip install -e lib/ -e server/

npm install --quiet --prefix server/src/morphocluster/server/frontend

flask db upgrade

flask add-user dev --password dev