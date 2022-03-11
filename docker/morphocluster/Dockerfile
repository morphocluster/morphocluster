FROM continuumio/miniconda3
WORKDIR /code

# supervisor is required to run multiple processes in parallel
RUN apt-get update --fix-missing && \
    apt-get install -y supervisor netcat rsync && \
    apt-get -qq --no-install-recommends install openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup SSH
RUN mkdir /var/run/sshd
RUN mkdir /root/.ssh && chmod 700 /root/.ssh

COPY docker/morphocluster/run.sh docker/wait-for environment*.yml docker/morphocluster/activate ./

COPY docker/morphocluster/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install all required packages at this stage to avoid reinstalling them later when the code changes
RUN --mount=type=cache,target=/opt/conda/pkg conda config --set notify_outdated_conda false && \
    conda config --set always_copy true && \
    if [ -f "environment.yml" ]; then \
        echo "Using environment.yml..." && \
        umask 0002 && conda env update -n base -f environment.yml;\
    elif [ -f "environment.default.yml" ]; then \
        echo "Using environment.default.yml..." && \
        umask 0002 && conda env update -n base -f environment.default.yml; \
    fi && \
    conda clean -y --all

# Install the application
COPY versioneer.py setup.py setup.cfg MANIFEST.in README.rst ./
COPY tests ./tests
COPY morphocluster ./morphocluster
COPY migrations ./migrations
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    echo Installing packages... && \
    pip install -e .

# Build frontend
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    cd morphocluster/frontend  && \
    echo Building frontend... && \
    echo NPM version: `npm --version` && \
    npm ci && \
    npm run build

#COPY --from=build_frontend /frontend/dist morphocluster/frontend/dist

CMD ["/code/run.sh"]