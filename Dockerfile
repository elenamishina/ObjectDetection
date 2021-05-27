FROM ubuntu:18.04

USER root

RUN apt update && \
	apt install -y --no-install-recommends \
	python3.7 \
	python3.7-dev \
	python3-setuptools \
	python3-pip \
	python3-opencv\
	libgl1-mesa-glx

COPY . /app
WORKDIR /app

RUN python3.7 -m pip install -U pip && python3.7 -m pip install -r requirements.txt
CMD streamlit run --server.port 8080 objectdetection.py
