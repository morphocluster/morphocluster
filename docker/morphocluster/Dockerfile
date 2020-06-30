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

COPY docker/morphocluster/run.sh docker/wait-for docker/morphocluster/environment.yml docker/morphocluster/activate ./

COPY docker/morphocluster/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install all required packages at this stage to avoid reinstalling them later when the code changes
RUN conda config --set notify_outdated_conda false && \
    conda env create -f environment.yml && \
    conda clean -y --all

# Install the application
COPY versioneer.py setup.py setup.cfg MANIFEST.in README.rst ./
COPY tests ./tests
COPY morphocluster ./morphocluster
COPY migrations ./migrations
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate morphocluster && \
    pip install -e .

# Build frontend
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate morphocluster && \
    cd morphocluster/frontend  && \
    npm ci && \
    npm run build

#COPY --from=build_frontend /frontend/dist morphocluster/frontend/dist

COPY docker/morphocluster/config_docker.py morphocluster/

CMD ["/code/run.sh"]