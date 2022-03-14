FROM mambaorg/micromamba
USER root

# Copy environment.default.yml and environment.yml (if found) to a temp location so we update the environment.
COPY environment*.yml /tmp/conda-tmp/
RUN --mount=type=cache,target=/opt/conda/pkg \
    if [ -f "/tmp/conda-tmp/environment.yml" ]; then \
    echo "Using environment.yml..." && \
    umask 0002 && micromamba install -n base -f /tmp/conda-tmp/environment.yml;\
    elif [ -f "/tmp/conda-tmp/environment.default.yml" ]; then \
    echo "Using environment.default.yml..." && \
    umask 0002 && micromamba install -n base -f /tmp/conda-tmp/environment.default.yml; fi \
    && rm -rf /tmp/conda-tmp