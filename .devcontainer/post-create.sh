#!/usr/bin/env sh

pip install -e lib/ -e server/

# Install frontend libraries
npm install --quiet --prefix /workspace/morphocluster/server/src/morphocluster/server/frontend

# Build frontend
npm run --prefix /workspace/morphocluster/server/src/morphocluster/server/frontend build

flask db upgrade

flask add-user dev --password dev