#!/usr/bin/env sh

pip install -e .

npm install --prefix morphocluster/frontend

flask add-user dev --password dev