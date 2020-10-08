FROM ubuntu:18.04

LABEL maintainer "Luca De Luigi <lucadeluigi91@gmail.com>"

SHELL ["/bin/bash", "-c"]

# install dependencies and tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip locales \
    libsm6 libxext6 libxrender-dev

# set locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# install python dependencies
RUN pip3 install numpy matplotlib jupyter opencv-python==3.4.2.16 opencv-contrib-python==3.4.2.16

# create dir for cvlab stuff
RUN mkdir -p /home/cvlab
WORKDIR "/home/cvlab"

# set default command
CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
