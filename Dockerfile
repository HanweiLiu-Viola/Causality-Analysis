# scipy-notebook already includes: numpy, pandas, scipy, matplotlib, seaborn,
# scikit-learn, statsmodels, numba, xarray, h5py, networkx, and Jupyter.
FROM quay.io/jupyter/scipy-notebook:python-3.12

# IDTxl routes all estimators through JIDT (Java Information Dynamics Toolkit)
# via JPype, so a headless JRE is required at runtime even for the Gaussian estimator.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*
USER ${NB_UID}
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# Install project-specific Python packages not bundled in the base image
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    mne==1.11.0 \
    mne-bids==0.18.0 \
    mne-connectivity==0.7.0 \
    frites==0.4.4 \
    h5netcdf==1.8.1 \
    netCDF4==1.7.4 \
    patsy==1.0.2 \
    beautifulsoup4==4.14.3 \
    pooch==1.9.0 \
    tqdm==4.67.3 \
    pybv==0.7.6

# IDTxl is not on PyPI; install from GitHub tag v1.6.0.
# Cython must be pre-installed because IDTxl's setup.py omits it from build deps.
RUN pip install --no-cache-dir Cython && \
    pip install --no-cache-dir \
        "idtxl @ git+https://github.com/pwollstadt/IDTxl.git@v1.6.0"

# Copy project source into the default notebook directory
COPY --chown=${NB_UID}:${NB_GID} src/ /home/${NB_USER}/work/src/
COPY --chown=${NB_UID}:${NB_GID} utils/ /home/${NB_USER}/work/utils/
COPY --chown=${NB_UID}:${NB_GID} scripts/ /home/${NB_USER}/work/scripts/
COPY --chown=${NB_UID}:${NB_GID} notebooks/ /home/${NB_USER}/work/notebooks/

# Non-interactive matplotlib backend
ENV MPLBACKEND=Agg
