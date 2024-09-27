# A mix of the Dockerfiles from OpenLM and Prismatic
# A bunch of stuff here might be unnecessary
ARG AWS_REGION

# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

# Sane Defaults
RUN apt-get update
RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    docker.io \
    ffmpeg \
    git \
    htop \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    libgl1 \
    libopenexr-dev \
    mesa-utils \
    freeglut3-dev \
    libsdl2-2.0-0 \
    python-pygame

# Remove the conda installed symlink for libcurl, which causes an error with curl.
# Fixes the following error:
# curl: /opt/conda/lib/libcurl.so.4: no version information available (required by curl)
RUN rm /opt/conda/lib/libcurl.so.4

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# IMPORTANT :: Uninstall & Reinstall Torch (Sagemaker CPU Core Bug)
RUN pip install --upgrade pip
RUN pip uninstall -y torch

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY requirements.txt /opt/ml/code/requirements.txt

RUN pip install -r /opt/ml/code/requirements.txt
RUN pip install packaging ninja
RUN pip uninstall transformer-engine -y
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip uninstall flash-attn -y
RUN pip install flash-attn==2.3.3
RUN pip uninstall huggingace_hub -y
RUN pip install --upgrade huggingface_hub
RUN pip uninstall boto3 -y
RUN pip install --upgrade boto3

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY . /opt/ml/code/

# revent sagemaker from installing requirements again.
RUN rm /opt/ml/code/requirements.txt

# Defines a script entrypoint
ENV SAGEMAKER_PROGRAM run.py
