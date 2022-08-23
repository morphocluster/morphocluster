FROM mambaorg/micromamba
WORKDIR /code
USER root

# supervisor is required to run multiple processes in parallel
RUN apt-get update --fix-missing && \
    apt-get install -y supervisor netcat rsync && \
    apt-get -qq --no-install-recommends install openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup SSH
RUN mkdir /var/run/sshd
RUN mkdir /root/.ssh && chmod 700 /root/.ssh

COPY environment*.yml /tmp

# Install all required packages at this stage to avoid reinstalling them later when the code changes
RUN --mount=type=cache,target=/opt/conda/pkg \
    micromamba config set always_copy true && \
    if [ -f "/tmp/environment.yml" ]; then \
    echo "Using environment.yml..." && \
    umask 0002 && micromamba install -n base -y -f /tmp/environment.yml;\
    elif [ -f "/tmp/environment.default.yml" ]; then \
    echo "Using environment.default.yml..." && \
    umask 0002 && micromamba install -n base -y -f /tmp/environment.default.yml; \
    fi

# Activate environment for subsequent RUN commands:
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install the application
COPY versioneer.py setup.py setup.cfg MANIFEST.in README.rst ./
COPY tests ./tests
COPY morphocluster ./morphocluster
COPY migrations ./migrations
RUN echo Installing packages... && \
    pip install -e .

# Build frontend
RUN cd morphocluster/frontend  && \
    echo Building frontend... && \
    echo NPM version: `npm --version` && \
    npm ci && \
    npm run build

#COPY --from=build_frontend /frontend/dist morphocluster/frontend/dist

COPY docker/wait-for docker/morphocluster/run.sh docker/morphocluster/activate ./
COPY docker/morphocluster/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN echo -e "export FLASK_APP=morphocluster\n" >> ~/.bashrc

CMD ["/code/run.sh"]