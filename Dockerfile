# syntax=docker/dockerfile:1
FROM ubuntu:22.04

# Update the package index and install git
RUN apt-get update && \
    apt-get install -y git curl
RUN mkdir /PhD_project

# Clone the GitHub repository
RUN git clone https://github.com/samcd22/PhD_project.git /PhD_project

#Set some environemnt variables we will need
ENV PATH="/build/miniconda3/bin:${PATH}"
ARG PATH="/build/miniconda3/bin:${PATH}"
RUN mkdir /build && \
    mkdir /build/.conda

#Install Python3.10 via miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /build/miniconda3 &&\
    rm -rf /Miniconda3-latest-Linux-x86_64.sh
WORKDIR /build
RUN conda install python=3.10

# Set the working directory
RUN pip install -r /PhD_project/requirements.txt
WORKDIR /PhD_project
RUN mkdir /PhD_project/data
RUN mkdir /PhD_project/results

RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libgl1-mesa-glx

RUN cd /PhD_project
# # Activate the Conda environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]