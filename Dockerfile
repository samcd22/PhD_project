# syntax=docker/dockerfile:1
FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

# Update the package index and install necessary packages
RUN apt-get update && \
    apt-get install -y git curl \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

# Clone the GitHub repository
RUN git clone https://github.com/samcd22/PhD_project.git /PhD_project

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /build/miniconda3 && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

# Create the Conda environment
RUN /build/miniconda3/bin/conda create -y -n myenv python=3.10

# Ensure pip is installed in the environment
RUN /build/miniconda3/bin/conda install -n myenv -y pip

# Clean Conda cache
RUN /build/miniconda3/bin/conda clean -afy

# Check if the environment was created correctly
RUN /build/miniconda3/bin/conda info -e

# Install Python dependencies inside the Conda environment
# RUN /build/miniconda3/envs/myenv/bin/pip install -r /PhD_project/requirements.txt

# Set the environment variables to activate the environment by default
ENV PATH="/build/miniconda3/envs/myenv/bin:$PATH"
ENV CONDA_DEFAULT_ENV=myenv

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Set the working directory
WORKDIR /PhD_project

# Create necessary directories
RUN mkdir /PhD_project/data /PhD_project/results

# Default command
CMD ["bash"]
