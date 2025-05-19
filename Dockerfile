# Base image con TensorFlow 2.0 GPU support (usa Python 3.6)
FROM tensorflow/tensorflow:2.0.0-gpu-py3

# Arreglar problema de clave GPG de NVIDIA y actualizar paquetes
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update

# Instalar librerías de sistema requeridas
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget

WORKDIR /workspace

# Copiar requirements completo
COPY ESRGAN/requirements.txt .

RUN pip install --upgrade pip

# Instalar dependencias (con versiones específicas para evitar conflictos)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir wandb==0.12.21 && \
    pip install --no-cache-dir tensorflow-hub==0.8.0

# Configurar variables de entorno
ENV PYTHONPATH=/workspace
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2

# Directorio para cachear modelos y datasets
RUN mkdir -p /workspace/cache /workspace/model /workspace/logs