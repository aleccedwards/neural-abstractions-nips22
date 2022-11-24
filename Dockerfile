FROM ubuntu:20.04

COPY . ./neural-abstraction

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update &&\
    apt-get install -y sudo curl vim python3 python3-pip tzdata libgmp3-dev &&\
    curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/20.04/install.sh | sudo bash &&\
    pip3 install -r neural-abstraction/requirements.txt
