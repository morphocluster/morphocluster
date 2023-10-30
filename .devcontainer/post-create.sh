#!/usr/bin/env sh

pip install -e .

npm install --quiet --prefix morphocluster/frontend

flask add-user dev --password dev