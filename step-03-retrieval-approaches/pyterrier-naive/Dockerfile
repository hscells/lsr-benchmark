FROM ubuntu:jammy


RUN apt-get update \
	&& apt-get install -y git openjdk-11-jdk python3 python3-pip

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

#	&& python3 -c "import pyterrier as pt; pt.init(version='${PYTERRIER_VERSION}', helper_version='${PYTERRIER_HELPER_VERSION}');" \
#	&& python3 -c "import pyterrier as pt; pt.init(version='${PYTERRIER_VERSION}', helper_version='${PYTERRIER_HELPER_VERSION}', boot_packages=['com.github.terrierteam:terrier-prf:-SNAPSHOT']);"

