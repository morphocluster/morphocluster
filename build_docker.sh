#!/usr/bin/env bash

# Set default for MORPHOCLUSTER_CUDA
: "${MORPHOCLUSTER_CUDA:=no}"

VERSION=$(git describe --tags --dirty)
TAG="$VERSION-cuda-$MORPHOCLUSTER_CUDA"

if [[ "${VERSION}" =~ "dirty" ]]; then
    echo "Working directory is dirty! Commit or stash changes first."
    exit
fi


docker build . -f docker/morphocluster/Dockerfile --progress plain --build-arg MORPHOCLUSTER_VERSION=$VERSION --build-arg MORPHOCLUSTER_CUDA=$MORPHOCLUSTER_CUDA -t morphocluster/morphocluster:$TAG