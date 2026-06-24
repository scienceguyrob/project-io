# =============================================================================
# project-io
# GitHub: https://github.com/scienceguyrob/project-io
#
# Builds a self-contained JupyterLab environment with all notebooks,
# challenges, datasets, and web pages included in the image.
#
# Local build & run:
#   docker build -f docker/Dockerfile -t project-io .
#   docker run --rm -p 8888:8888 project-io
#
# Then open: http://localhost:8888
# =============================================================================

FROM python:3.13-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Create a generic non-root user for running JupyterLab.
# Using a dedicated user rather than root is a standard security practice
# for Docker containers. The username 'notebook' is internal to the image
# and has no relation to the username of whoever is running the container.
# ---------------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash --uid 1000 notebook

# ---------------------------------------------------------------------------
# Python dependencies — pinned to match the project specification
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    numpy==2.2.4 \
    pandas==2.2.3 \
    matplotlib==3.10.0 \
    seaborn==0.13.2 \
    scipy==1.15.2 \
    scikit-learn==1.6.1 \
    ipywidgets==8.1.7 \
    ipython==8.30.0 \
    imbalanced-learn==0.14.1 \
    jupyterlab==4.3.6 \
    notebook==7.3.3 \
    ipympl==0.9.7

# ---------------------------------------------------------------------------
# Project files
# All content is copied into /home/notebook — the internal working directory
# for this container. This path is entirely self-contained within the image
# and has no relation to the home directory of whoever runs the container.
# JupyterLab is configured to serve files from here, which is what makes
# URLs like http://localhost:8888/files/index.html resolve correctly.
# ---------------------------------------------------------------------------
WORKDIR /home/notebook

# Copy the full project structure into the image
COPY --chown=notebook:notebook index.html         ./index.html
COPY --chown=notebook:notebook web/               ./web/
COPY --chown=notebook:notebook Notebook_0/        ./Notebook_0/
COPY --chown=notebook:notebook Notebook_1/        ./Notebook_1/
COPY --chown=notebook:notebook Notebook_2/        ./Notebook_2/
COPY --chown=notebook:notebook Notebook_3/        ./Notebook_3/
COPY --chown=notebook:notebook Notebook_4/        ./Notebook_4/
COPY --chown=notebook:notebook Notebook_5/        ./Notebook_5/
COPY --chown=notebook:notebook Notebook_6/        ./Notebook_6/
COPY --chown=notebook:notebook Notebook_6_Bonus/  ./Notebook_6_Bonus/
COPY --chown=notebook:notebook Notebook_7/        ./Notebook_7/
COPY --chown=notebook:notebook Notebook_7_Bonus/  ./Notebook_7_Bonus/
COPY --chown=notebook:notebook Notebook_8/        ./Notebook_8/
COPY --chown=notebook:notebook Notebook_8_Bonus/  ./Notebook_8_Bonus/
COPY --chown=notebook:notebook Notebook_9/        ./Notebook_9/
COPY --chown=notebook:notebook Notebook_10/       ./Notebook_10/
COPY --chown=notebook:notebook Notebook_11/       ./Notebook_11/
COPY --chown=notebook:notebook Notebook_12/       ./Notebook_12/
COPY --chown=notebook:notebook Notebook_13/       ./Notebook_13/

# ---------------------------------------------------------------------------
# JupyterLab configuration
# No token / no password — suitable for local use.
# root_dir points to /home/notebook so the JupyterLab file browser
# opens at the project root, where all notebooks and web pages live.
# ---------------------------------------------------------------------------
RUN mkdir -p /home/notebook/.jupyter && \
    echo "c.ServerApp.token = ''" >> /home/notebook/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /home/notebook/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.open_browser = False" >> /home/notebook/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /home/notebook/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.root_dir = '/home/notebook'" >> /home/notebook/.jupyter/jupyter_server_config.py && \
    echo "c.ContentsManager.allow_hidden = True" >> /home/notebook/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.disable_check_xsrf = True" >> /home/notebook/.jupyter/jupyter_server_config.py && \
    chown -R notebook:notebook /home/notebook/.jupyter

# ---------------------------------------------------------------------------
# Switch to the non-root user — never run JupyterLab as root
# ---------------------------------------------------------------------------
USER notebook

# ---------------------------------------------------------------------------
# Expose JupyterLab port and start
# ---------------------------------------------------------------------------
EXPOSE 8888

CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888"]