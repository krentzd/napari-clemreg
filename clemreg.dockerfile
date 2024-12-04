FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 AS clemreg

ARG DEBIAN_FRONTEND=noninteractive

# https://github.com/napari/napari/blob/main/dockerfile
# install python resources + graphical libraries used by qt and vispy
RUN apt-get update && \
    apt-get install -qqy  \
        build-essential \
        python3.9 \
        python3-pip \
        git \
        mesa-utils \
        x11-utils \
        libegl1-mesa \
        libopengl0 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libfontconfig1 \
        libxrender1 \
        libdbus-1-3 \
        libxkbcommon-x11-0 \
        libxi6 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xinput0 \
        libxcb-xfixes0 \
        libxcb-shape0 \
        wget \
        && apt-get clean

# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p ~/.conda && \
    bash miniconda.sh -b -p /opt/miniconda3 && \
    rm -f miniconda.sh

ENV PATH="/opt/miniconda3/bin:$PATH"
ARG PATH="/opt/miniconda3/bin:$PATH"

# Create conda env to ensure controlled Python version
RUN conda create --yes -n clemreg_env python=3.9

# Ensure all future shells are in the conda env
RUN echo "conda activate clemreg_env" >> ~/.bashrc

SHELL ["conda", "run", "-n", "clemreg_env", "/bin/bash", "-c"]

# Install Napari
RUN pip install --upgrade pip && \
    conda install -y -c conda-forge pyqt && \
    pip install napari && \
    pip install napari-clemreg && \
    conda clean -afy && \
    pip cache purge

# Set the default Conda environment
ENV CONDA_DEFAULT_ENV=clemreg_env
ENV PATH="/opt/miniconda3/envs/clemreg_env/bin:$PATH"

ENTRYPOINT ["bash", "-c", "source activate clemreg_env && napari"]


#########
# If X11 not available or issues, use Xpra
#########

FROM clemreg AS clemreg_xpra

# Install stuff for Xpra server
RUN apt-get update && apt-get install -y wget gnupg2 apt-transport-https \
    software-properties-common ca-certificates && \
    wget -O "/usr/share/keyrings/xpra.asc" https://xpra.org/xpra.asc && \
    wget -O "/etc/apt/sources.list.d/xpra.sources" https://raw.githubusercontent.com/Xpra-org/xpra/master/packaging/repos/jammy/xpra.sources

RUN apt-get update && \
    apt-get install -yqq \
        xpra \
        xvfb \
        xterm \
        sshfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup envs and ports for Xpra
ENV DISPLAY=:100
ENV XPRA_PORT=9876
ENV XPRA_START="napari"
ENV XPRA_EXIT_WITH_CLIENT="yes"
ENV XPRA_XVFB_SCREEN="1920x1080x24+32"
EXPOSE 9876

CMD echo "Launching napari on Xpra. Connect via http://localhost:$XPRA_PORT or $(hostname -i):$XPRA_PORT"; \
    xpra start \
    --bind-tcp=0.0.0.0:$XPRA_PORT \
    --html=on \
    --start="$XPRA_START" \
    --exit-with-client="$XPRA_EXIT_WITH_CLIENT" \
    --daemon=no \
    --xvfb="/usr/bin/Xvfb +extension Composite -screen 0 $XPRA_XVFB_SCREEN -nolisten tcp -noreset" \
    --pulseaudio=no \
    --notifications=no \
    --bell=no \
    $DISPLAY

# Clear the entrypoint to avoid any weird inherit? Not sure, but Napari people did this
ENTRYPOINT []