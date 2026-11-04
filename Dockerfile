# Reproducible analysis environment for dirconn.
#
# The library itself is used through `pip install dirconn[extras]`. This image has one job: to
# freeze the whole scientific stack for real-data analyses, so that their numbers can be computed
# again years later, on a workstation (Docker) or on an HPC system (Apptainer). It is an
# environment, not a workflow: there is no entrypoint and no pipeline inside; analysis scripts
# and notebooks are mounted into /work and run there.
#
# Build:  python scripts/build_image.py          (passes the git SHA, date and version)
# Use:    see "Reproducible Docker environment" in README.md

ARG PYTHON_IMAGE=python:3.11.16-slim-bookworm
FROM ${PYTHON_IMAGE}

# Thread counts change the order of floating-point sums and therefore results; one thread by
# default keeps runs repeatable. Parallelise over tasks (subjects, surrogates) with joblib and
# explicit seeds instead, or override for speed: docker run -e OMP_NUM_THREADS=8 ...
# MPLBACKEND: no display in a container. MNE_DATA: where the MNE `sample` dataset is mounted
# (it is never downloaded without being asked). The image is read-only under Apptainer, so
# nothing may be written into it at run time; caches go to the home directory or /tmp.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLBACKEND=Agg \
    MNE_DATA=/mne_data \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git: the analysis repositories mounted into /work record their commit (dirconn.protocol); they
# belong to another user than the container's, which git refuses unless the directory is trusted.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system safe.directory '*'

COPY . /opt/dirconn-src

# Extras: everything except `gpu` (torch is measured separately, scripts/gpu_check.py) and
# `plot-extra` (3D needs a display; surface maps are drawn on the host from exported results).
# pytest, nbformat and packaging run the test suite inside the image. The notebooks are generated
# from examples/ and are never committed, so they are made here.
RUN pip install --no-cache-dir -c /opt/dirconn-src/constraints/docker.txt \
        "/opt/dirconn-src[connectivity,pcmci,stats,parallel,speed,plot]" \
        jupyterlab pytest nbformat packaging \
 && pip freeze > /opt/environment-lock.txt \
 && python /opt/dirconn-src/scripts/make_notebooks.py --out /opt/dirconn/tutorials \
 && python -c "import dirconn, mne, mne_connectivity, tigramite, pycirclize; print('dirconn', dirconn.__version__)"

# A normal user: with a bind-mounted /work, a root container writes files the host user may not
# own. UID 1000 is the usual first account on Linux; pass --user to docker run when it is not.
# The source tree is handed over so that pytest can write its cache there.
RUN useradd --create-home --uid 1000 dirconn \
 && mkdir -p /work /data /mne_data \
 && chown -R dirconn:dirconn /work /data /mne_data /opt/dirconn-src /opt/dirconn
USER dirconn
WORKDIR /work
EXPOSE 8888
CMD ["bash"]

# Provenance, deliberately last: an ARG invalidates every layer below it, so a new GIT_SHA does
# not trigger a full rebuild. The installed copy is not a git checkout, so the commit it was built
# from is handed to dirconn here (code_version), and every result's provenance names the image.
ARG GIT_SHA=unknown
ARG GIT_DIRTY=1
ARG BUILD_DATE=unknown
ARG VERSION=unknown
ARG IMAGE=unknown
ENV DIRCONN_GIT_COMMIT=${GIT_SHA} \
    DIRCONN_GIT_DIRTY=${GIT_DIRTY} \
    DIRCONN_IMAGE=${IMAGE}
LABEL org.opencontainers.image.title="dirconn" \
      org.opencontainers.image.description="Reproducible analysis environment for dirconn (directed connectivity)" \
      org.opencontainers.image.source="https://github.com/HanweiLiu-Viola/dirconn" \
      org.opencontainers.image.licenses="BSD-3-Clause" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.version="${VERSION}"
