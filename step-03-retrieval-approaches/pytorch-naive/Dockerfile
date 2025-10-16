FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ADD requirements.txt /tmp/

RUN apt-get update && apt-get install -y git && python3 -m pip install --no-cache -r /tmp/requirements.txt

ADD search-pytorch-index.py /search-pytorch-index.py
