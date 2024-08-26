FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 7
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 7

WORKDIR /app

RUN python3 -m venv Insam_cls && \
    . Insam_cls/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["/bin/bash"]