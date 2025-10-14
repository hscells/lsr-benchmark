FROM ubuntu:jammy


RUN apt-get update \
	&& apt-get install -y git openjdk-21-jdk python3 python3-pip

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

RUN pip3 install faiss-cpu
RUN pip3 install torch pip --index-url https://download.pytorch.org/whl/cpu

ADD run-pyserini-lsr.py /run-pyserini-lsr.py

