# Build frontend
# FROM node:10 AS build_frontend
# WORKDIR /frontend
# COPY morphocluster/frontend .
# RUN cd /frontend  && \
#     npm ci && \
#     npm run build

FROM continuumio/miniconda3
WORKDIR /code

# supervisor is required to run multiple processes in parallel
RUN apt-get update --fix-missing && \
    apt-get install -y supervisor netcat && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY docker/run.sh docker/wait-for docker/environment.yml docker/activate ./

# Install all required packages at this stage to avoid reinstalling them later when the code changes
RUN conda config --set notify_outdated_conda false && \
    conda env create -f environment.yml && \
    conda clean -y --all

# Install prerequisites
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate morphocluster && \
    pip install gunicorn

# Install the application
COPY versioneer.py setup.py setup.cfg MANIFEST.in README.md ./
COPY tests ./tests
COPY morphocluster ./morphocluster
COPY migrations ./migrations
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate morphocluster && \
    pip install -e .

#COPY --from=build_frontend /frontend/dist morphocluster/frontend/dist
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY docker/config_docker.py morphocluster/

CMD ["/code/run.sh"]