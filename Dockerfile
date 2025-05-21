# Base image con TensorFlow 2.11.0 y Python 3.9
FROM tensorflow/tensorflow:2.11.0-gpu

# Establecer variables de entorno para evitar interacciones durante la instalación
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Bogota

# Arreglar problema de clave GPG y actualizar paquetes
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Actualizar repositorios
RUN apt-get update

# Instalar Python 3.9 y dependencias necesarias
RUN apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    python3-setuptools \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Asegurar que python3 y pip3 apunten a Python 3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Verificar versión de Python
RUN python --version && pip --version

WORKDIR /workspace

# Copiar requirements.txt
COPY ESRGAN/requirements.txt .

# Actualizar pip y setuptools
RUN pip install --upgrade pip setuptools wheel

# Instalar dependencias para TensorFlow 2.11.0
RUN pip install --no-cache-dir \
    absl-py==1.4.0 \
    attrs==23.1.0 \
    cachetools==5.3.1 \
    numpy==1.23.5 \
    scipy==1.10.1 \
    tensorflow==2.11.0 \
    tensorflow-datasets==4.8.3 \
    tensorflow-estimator==2.11.0 \
    tensorflow-io-gcs-filesystem==0.31.0 \
    tensorflow-metadata==1.13.1 \
    tensorflow-hub==0.13.0 \
    PyYAML==6.0 \
    pillow==9.5.0 \
    wandb==0.15.4 \
    tqdm==4.65.0

# Instalar dependencias adicionales
RUN pip install --no-cache-dir \
    protobuf==3.19.6 \
    tensorboard==2.11.2 \
    grpcio==1.54.2 \
    h5py==3.8.0

# Configurar variables de entorno
ENV PYTHONPATH=/workspace
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2

# Directorio para cachear modelos y datasets
RUN mkdir -p /workspace/cache /workspace/model /workspace/logs

# Verificar instalación de TensorFlow
RUN python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU'))>0)"