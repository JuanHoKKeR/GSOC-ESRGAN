#!/bin/bash

# Script para iniciar el entrenamiento de ESRGAN con microscopía

# Crear directorios necesarios
mkdir -p logs model cache

# Construir la imagen Docker
echo "Construyendo imagen Docker..."
docker-compose build

# Iniciar contenedor
echo "Iniciando contenedor..."
docker-compose up -d

# Ejecutar comando de entrenamiento dentro del contenedor
echo "Ejecutando entrenamiento..."
docker exec -it esrgan-container bash -c "cd /workspace && \
    python train_microscopy_meta_info.py --hr_meta_file 'datasets/paired_meta/paired_hr_meta.txt' --lr_meta_file 'datasets/paired_meta/paired_lr_meta.txt' --model_dir '/workspace/model' --log_dir '/workspace/logs' --kimianet_weights '/workspace/model-kimianet/KimiaNetKerasWeights.h5' --batch_size 16 --wandb_name 'microscopy-training' --phase 'phase1_phase2' -v"

# python train_microscopy_meta_info.py --hr_meta_file 'datasets/paired_meta/paired_hr_meta.txt' --lr_meta_file 'datasets/paired_meta/paired_lr_meta.txt' --model_dir '/workspace/model' --log_dir '/workspace/logs' --batch_size 8 --wandb_name 'microscopy-training-kimianet' --phase 'phase1_phase2' -v

#    python train_microscopy_meta_info.py --hr_meta_file 'datasets/paired_meta/paired_hr_meta.txt' --lr_meta_file 'datasets/paired_meta/paired_lr_meta.txt' --model_dir '/workspace/model' --log_dir '/workspace/logs' --kimianet_weights '/workspace/model-kimianet/KimiaNetKerasWeights.h5' --pretrained_model '/workspace/model-esrgantf2/saved_model.pb' --batch_size 16 --wandb_name 'microscopy-training' --phase 'phase1_phase2' -v
# python train_microscopy_meta_info.py --hr_meta_file 'datasets/paired_meta/paired_hr_meta.txt' --lr_meta_file 'datasets/paired_meta/paired_lr_meta.txt' --model_dir '/workspace/model' --log_dir '/workspace/logs' --batch_size 16 --wandb_name 'microscopy-training-64to256_phase2' --phase 'phase2' -v


# python train_microscopy_meta_info.py --hr_meta_file 'datasets/paired_meta/paired_hr_meta.txt' --lr_meta_file 'datasets/paired_meta/paired_lr_meta.txt' --model_dir '/workspace/model' --log_dir '/workspace/logs' --kimianet_weights '/workspace/model-kimianet/KimiaNetKerasWeights.h5' --batch_size 16 --wandb_project 'ESRGAN-Microscopy-X2'  --wandb_name 'microscopy-training-128to256_phase2' --phase 'phase1_phase2' -v


# Usar este comando para verificar logs
echo "Puedes verificar logs con:"
echo "docker logs -f esrgan-container"

# Para detener el contenedor cuando termine
echo "Para detener el contenedor cuando termine:"
echo "docker-compose down"