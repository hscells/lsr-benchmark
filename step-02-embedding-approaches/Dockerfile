FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ADD requirements.txt /tmp/

RUN set -x \
    && python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install --no-cache -r /tmp/requirements.txt \
    && rm -rf ./build ./*.egg-info

ENV HF_HUB_OFFLINE=1

ADD lightning-ir.py /lightning-ir.py
