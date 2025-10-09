# Take from the docker container of seismic. it does not seems to require any change. 
# docker build -t mam10eks/kannolo:0.0.1 .
FROM ubuntu:22.04

RUN apt-get update \
	&& apt-get install -y python3 python3-pip git libopenmpi-dev openmpi-bin \
	&& pip3 install maturin \
	&& apt-get clean \
	&& pip3 cache purge

ADD requirements.txt /tmp/requirements.txt

RUN rustup install nightly \
    && rustup default nightly

RUN pip3 install -r /tmp/requirements.txt \
	&& rm -Rf /tmp/requirements.txt \
	&& pip3 cache purge
