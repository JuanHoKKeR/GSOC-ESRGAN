# Base image with TensorFlow 2.0 GPU support (comes with Python 3.6)
FROM tensorflow/tensorflow:2.0.0-gpu-py3

# Fix NVIDIA GPG key issue and update packages
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update

# Install required system libraries
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

WORKDIR /workspace

# Copy requirements file
COPY ESRGAN/requirements.txt .


