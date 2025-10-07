FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN set -x \
    && apt update \
    && apt install -y git python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt /tmp/

RUN set -x \
    && python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install --no-cache -r /tmp/requirements.txt \
    && rm -rf ./build ./*.egg-info

ENV HF_HUB_OFFLINE=1

ADD lightning-ir.py /lightning-ir.py
