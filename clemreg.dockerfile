# napari with napari-clemreg, including the default (Segment-Flow) EM
# segmentation backend. Build from the repository root:
#
#   docker build -t clemreg --target clemreg -f clemreg.dockerfile .
#
# Segment-Flow builds MitoNet's own environment, with its own PyTorch, on
# first use, so no CUDA base image is needed: `docker run --gpus=all` still
# gives it the GPU.
FROM ubuntu:22.04 AS clemreg

ARG DEBIAN_FRONTEND=noninteractive

# Libraries Qt, vispy and open3d need (https://github.com/napari/napari/blob/main/dockerfile),
# and a C++ compiler: probreg has no Python 3.11 wheel, so pip builds it.
RUN apt-get update && \
    apt-get install -qqy --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        wget \
        mesa-utils \
        x11-utils \
        libegl1-mesa \
        libopengl0 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libusb-1.0-0 \
        libfontconfig1 \
        libxrender1 \
        libdbus-1-3 \
        libxkbcommon-x11-0 \
        libxi6 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xinerama0 \
        libxcb-xinput0 \
        libxcb-xfixes0 \
        libxcb-shape0 && \
    rm -rf /var/lib/apt/lists/*

# Miniforge: Segment-Flow needs `conda` to build the model's environment.
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
        -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh
ENV PATH="/opt/conda/bin:$PATH"

# Python, plus Nextflow and a JDK for Segment-Flow.
RUN conda create -y -n clemreg -c conda-forge -c bioconda python=3.11 nextflow "openjdk=21" && \
    conda clean -afy
ENV PATH="/opt/conda/envs/clemreg/bin:$PATH"

COPY . /opt/napari-clemreg
RUN pip install --no-cache-dir "/opt/napari-clemreg[segment-flow]" pyqt6

CMD ["napari"]


#########
# Without an X server (e.g. on macOS), use Xpra and open napari in a browser:
#   docker build -t clemreg_xpra --target clemreg_xpra -f clemreg.dockerfile .
#   docker run --rm -p 9876:9876 clemreg_xpra
#########

FROM clemreg AS clemreg_xpra

RUN apt-get update && \
    apt-get install -qqy --no-install-recommends gnupg2 apt-transport-https && \
    wget -O "/usr/share/keyrings/xpra.asc" https://xpra.org/xpra.asc && \
    wget -O "/etc/apt/sources.list.d/xpra.sources" \
        https://raw.githubusercontent.com/Xpra-org/xpra/master/packaging/repos/jammy/xpra.sources && \
    apt-get update && \
    apt-get install -qqy xpra xvfb xterm && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:100
ENV XPRA_PORT=9876
ENV XPRA_XVFB_SCREEN="1920x1080x24+32"
EXPOSE 9876

CMD echo "Launching napari on Xpra. Connect via http://localhost:$XPRA_PORT"; \
    xpra start \
    --bind-tcp=0.0.0.0:$XPRA_PORT \
    --html=on \
    --start="napari" \
    --exit-with-client=yes \
    --daemon=no \
    --xvfb="/usr/bin/Xvfb +extension Composite -screen 0 $XPRA_XVFB_SCREEN -nolisten tcp -noreset" \
    --pulseaudio=no \
    --notifications=no \
    --bell=no \
    $DISPLAY
