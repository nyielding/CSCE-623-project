FROM tensorflow/tensorflow:latest-gpu-jupyter as develop

# Install system libraries for python packages
RUN apt-get update &&  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        graphviz \
        python3-tk \
        cmake \
        # required for cv2
        libgl1-mesa-glx \
        xvfb \
        wget \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install other python library requirements
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# Update jupyter, install JupyterLab
RUN pip install jupyter -U && pip install jupyterlab && pip install -U ipython>=7.20 && pip install -U jedi>=0.18
