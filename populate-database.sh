#!/usr/bin/env sh

flask user add test-user --password test-user

flask dataset create test-dataset test-user --objects tests/data/objects.zip --features tests/data/features.h5