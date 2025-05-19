#!/bin/bash

# Script para monitorear el progreso del entrenamiento

# Iniciar TensorBoard en segundo plano
echo "Iniciando TensorBoard..."
docker exec -d esrgan-container bash -c "tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006"

echo "TensorBoard iniciado en http://localhost:6006"
echo "Presiona Ctrl+C para salir del monitor de logs"

# Mostrar los logs en tiempo real
docker exec -it esrgan-container bash -c "tail -f /workspace/logs/phase1/*"