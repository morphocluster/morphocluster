# Build the micromamba environment file
FROM python:3 as build-env

# Configure CUDA environment
ARG MORPHOCLUSTER_CUDA=no

# Build environment.yml for main image
WORKDIR /build-env
COPY docker/morphocluster/build-env /build-env
COPY environment*.yml ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python build.py

# Build the frontend
FROM node:18-alpine as build-frontend
WORKDIR /frontend
COPY morphocluster/frontend /frontend

# --openssl-legacy-provider to support some encryption algorithms needed by npm (md4) in OpenSSL 3.0.0
# See https://github.com/vuejs/vue-cli/issues/6770
RUN export NODE_OPTIONS=--openssl-legacy-provider && \
    echo Building frontend... && \
    echo NPM version: `npm --version` && \
    npm ci && \
    npm run build

# Main image
FROM mambaorg/micromamba
WORKDIR /code
USER root

# TODO: Drop SSH
# supervisor is required to run multiple processes in parallel
RUN apt-get update --fix-missing && \
    apt-get install -y supervisor netcat rsync && \
    apt-get -qq --no-install-recommends install openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Image metadata
ARG MORPHOCLUSTER_VERSION=unversioned

# Labels
# see https://github.com/opencontainers/image-spec/blob/main/annotations.md
LABEL org.opencontainers.image.title=MorphoCluster
LABEL org.opencontainers.image.version=${MORPHOCLUSTER_VERSION}
LABEL org.opencontainers.image.source=https://github.com/morphocluster/morphocluster/

# Setup SSH
RUN mkdir /var/run/sshd
RUN mkdir /root/.ssh && chmod 700 /root/.ssh

COPY --from=build-env /build-env/environment.yml /tmp

# Install all required packages at this stage to avoid reinstalling them later when the code changes
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba config set always_copy true && \
    umask 0002 && micromamba install -n base -y -f /tmp/environment.yml;

# Activate environment for subsequent RUN commands:
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install the application
COPY versioneer.py setup.py setup.cfg MANIFEST.in README.rst ./
COPY tests ./tests
COPY morphocluster ./morphocluster
COPY migrations ./migrations
RUN echo Installing packages... && \
    pip install -e .

COPY --from=build-frontend /frontend/dist morphocluster/frontend/dist

COPY docker/wait-for docker/morphocluster/run.sh docker/morphocluster/activate ./
COPY docker/morphocluster/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENV FLASK_APP=morphocluster

# Enable bash completion
RUN _FLASK_COMPLETE=bash_source flask > ~/.flask-complete.bash && \
    echo -e ". ~/.flask-complete.bash\n" >> ~/.bashrc && \
    _MORPHOCLUSTER_COMPLETE=bash_source morphocluster > ~/.morphocluster-complete.bash && \
    echo -e ". ~/.morphocluster-complete.bash\n" >> ~/.bashrc

CMD ["/code/run.sh"]