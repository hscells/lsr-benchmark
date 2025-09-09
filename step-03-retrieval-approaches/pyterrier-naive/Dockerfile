FROM ubuntu:jammy


RUN apt-get update \
	&& apt-get install -y git openjdk-11-jdk python3 python3-pip

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

ENV PYTERRIER_VERSION=5.11
ENV PYTERRIER_HELPERE_VERSION=0.0.8

python3 -c "import pyterrier as pt; pt.init(version='${PYTERRIER_VERSION}', helper_version='${PYTERRIER_HELPER_VERSION}');"

ADD run-pyterrier.py /run-pyterrier.py

