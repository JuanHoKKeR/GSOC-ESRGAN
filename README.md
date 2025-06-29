# ESRGAN for Histopathology Super-Resolution

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![TensorFlow 2.11](https://img.shields.io/badge/TensorFlow-2.11-orange.svg)](https://tensorflow.org/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Una implementación modernizada de Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN) específicamente optimizada para imágenes de histopatología de cáncer de mama. Este proyecto forma parte de un **Trabajo de Grado** enfocado en super-resolución para aplicaciones médicas.

## 🎯 **Objetivo del Proyecto**

Desarrollar y evaluar modelos de super-resolución basados en ESRGAN para mejorar la calidad de imágenes de microscopia histopatológica, facilitando el análisis automatizado y el diagnóstico asistido por computadora.

## ✨ **Características Principales**

- **🔬 Especializado en Histopatología**: Optimizado para imágenes de cáncer de mama
- **🧠 Discriminador KimiaNet**: Utiliza pesos preentrenados de KimiaNet para mejor discriminación en imágenes médicas
- **📊 Múltiples Escalas**: Modelos entrenados para factores 2×, 4×, 8× y 16×
- **🐳 Dockerizado**: Entorno completamente contenedorizado para reproducibilidad
- **⚡ Modernizado**: Actualizado a Python 3.9 y TensorFlow 2.11.0
- **📈 Sistema de Evaluación Completo**: Métricas PSNR, SSIM, MS-SSIM, MSE e índice perceptual

## 🔄 **Diferencias con el Proyecto Original**

Este repositorio está basado en [captain-pool/GSOC](https://github.com/captain-pool/GSOC/tree/master/E2_ESRGAN) pero incluye mejoras significativas:

| Aspecto | Original | Esta Implementación |
|---------|----------|-------------------|
| **Python** | 3.6 | 3.9 |
| **TensorFlow** | TensorFlow-GPU (legacy) | TensorFlow 2.11.0 |
| **Dominio** | Imágenes generales | Histopatología específica |
| **Discriminador** | VGG estándar | DenseNet121 + KimiaNet |
| **Entorno** | Manual | Docker + Docker Compose |
| **Evaluación** | Básica | Sistema comprehensive |

## 🚀 **Inicio Rápido con Docker**

### Prerequisitos
- Docker y Docker Compose instalados
- GPU NVIDIA con drivers compatibles (opcional pero recomendado)
- NVIDIA Container Toolkit para soporte GPU

### 1. Clonar el Repositorio
```bash
git clone https://github.com/JuanHoKKeR/GSOC-ESRGAN.git
cd GSOC-ESRGAN
```

### 2. Preparar el Dataset
Organiza tu dataset con la siguiente estructura:
```
data/
├── train/
│   ├── hr/          # Imágenes de alta resolución
│   └── lr/          # Imágenes de baja resolución  
├── validation/
│   ├── hr/
│   └── lr/
└── test/
    ├── hr/
    └── lr/
```

### 3. Configurar KimiaNet (Opcional)
Para usar el discriminador DenseNet+KimiaNet:

```bash
# Crear directorio para pesos
mkdir -p model-kimianet/

# Descargar pesos de KimiaNet
# Los pesos pueden obtenerse de: https://github.com/KimiaLabMayo/KimiaNet/blob/main/KimiaNet_Weights/weights/KimiaNetKerasWeights.h5
# Archivo: KimiaNetKerasWeights.h5
```

### 4. Configurar el Entrenamiento
Edita `ESRGAN/config/config.yaml`:

```yaml
# Configuración básica
batch_size: 4
dataset:
  hr_dimension: 256    # Resolución objetivo
  lr_dimension: 128    # Resolución de entrada
  name: "MiDataset"
  scale_method: bicubic

# Discriminador
discriminator:
  type: densenet                                    # densenet | vgg | optimized_vgg
  kimianet_weights: /workspace/model-kimianet/KimiaNetKerasWeights.h5

# Generador (modificar según memoria disponible)
generator:
  num_features: 32     # Reducir si hay problemas de memoria
  trunk_size: 11       # Reducir para modelos más ligeros
  growth_channel: 32   # Reducir para optimización
```

### 5. Ejecutar con Docker
```bash
# Construir contenedor
docker-compose build

# Ejecutar entrenamiento
docker-compose run --rm esrgan python train_microscopy_meta_info.py \
    --hr_meta_file datasets/paired_meta/paired_hr_meta.txt \
    --lr_meta_file datasets/paired_meta/paired_lr_meta.txt \
    --model_dir ./model \
    --log_dir ./logs \
    --config config/config.yaml
```

## 📊 **Modelos Entrenados**

### Arquitecturas Implementadas

| Modelo | Escala | Input → Output | Parámetros | Optimización |
|--------|--------|----------------|------------|--------------|
| 64→128 | ×2 | 64×64 → 128×128 | Estándar (32,11,32) | - |
| 64→256 | ×4 | 64×64 → 256×256 | Estándar (32,11,32) | - |
| 128→256 | ×2 | 128×128 → 256×256 | Estándar (32,11,32) | - |
| 128→512 | ×4 | 128×128 → 512×512 | Estándar (32,11,32) | - |
| 256→512 | ×2 | 256×256 → 512×512 | Estándar (32,11,32) | - |
| 512→1024 | ×2 | 512×512 → 1024×1024 | **Optimizado (16,6,16)** | Memoria |
| 64→1024 | ×16 | 64×64 → 1024×1024 | Estándar (32,11,32) | - |
| 128→1024 | ×8 | 128×128 → 1024×1024 | **Modificado (64,23,64)** | Gradientes |
| 256→1024 | ×4 | 256×256 → 1024×1024 | **Modificado (40,11,40)** | Gradientes |

*Parámetros: (num_features, trunk_size, growth_channel)*

### Resultados Preliminares (Promedio en dataset de validación)

| Modelo | PSNR (dB) | SSIM | MS-SSIM | Tiempo GPU (ms) |
|--------|-----------|------|---------|-----------------|
| 128→256 | 29.45±2.1 | 0.847±0.05 | 0.821±0.06 | ~15 |
| 256→512 | 27.82±1.8 | 0.824±0.04 | 0.798±0.05 | ~45 |
| 512→1024 | 25.93±1.6 | 0.789±0.06 | 0.765±0.07 | ~180 |

*Evaluado en ~122k imágenes de validación de histopatología de cáncer de mama*

## 🛠️ **Personalización de Arquitectura**

### Modificar Parámetros del Generador
En `config.yaml`:
```yaml
generator:
  num_features: 32      # Características base (16-64)
  trunk_size: 11        # Bloques RRDB (6-23)  
  growth_channel: 32    # Canales de crecimiento (16-64)
```

### Configurar Discriminador
```yaml
discriminator:
  type: densenet                    # Recomendado para histopatología
  # type: vgg                       # Estándar
  # type: optimized_vgg             # Para memoria limitada
  
  kimianet_weights: /path/to/KimiaNetKerasWeights.h5 # Si se omiten los pesos de la KimiaNet se hara uso de la DenseNet por defecto
  num_features: 64                  # Solo para VGG
```

## 📈 **Sistema de Evaluación**

El proyecto incluye un sistema completo de evaluación:

### Métricas Implementadas
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index  
- **MS-SSIM**: Multi-Scale SSIM
- **MSE**: Mean Squared Error
- **Índice Perceptual**: Basado en KimiaNet

### Evaluación en Color y Escala de Grises
```bash
# Evaluar modelo específico
python evaluate_model_validation.py \
    --model_path ./models/esrgan_256to512 \
    --model_name "256to512" \
    --lr_meta_file datasets/validation/256to512/paired_lr_meta.txt \
    --hr_meta_file datasets/validation/256to512/paired_hr_meta.txt
```

### Análisis Visual de Diferencias
```bash
# Análisis comprehensive de diferencias
python visual_difference_analyzer.py \
    --lr_image path/to/lr_image.jpg \
    --hr_image path/to/hr_image.jpg \
    --model_path ./models/esrgan_model
```

## 📚 **Dataset y Entrenamiento**

### Preparación del Dataset
1. **Imágenes de entrenamiento**: ~440k imágenes de histopatología
2. **Validación**: ~122k imágenes
3. **Formato**: JPEG, RGB
4. **Organización**: Archivos `meta_info` para rutas de imágenes

### Archivos Meta-Info
Crea archivos de texto con rutas a las imágenes:
```
# paired_hr_meta.txt
/data/train/hr/image001.png
/data/train/hr/image002.png
...

# paired_lr_meta.txt  
/data/train/lr/image001.png
/data/train/lr/image002.png
...
```

## 🐳 **Configuración Docker**

### Variables de Entorno
```bash
# En docker-compose.yml o .env
NVIDIA_VISIBLE_DEVICES=all
TF_FORCE_GPU_ALLOW_GROWTH=true
```

### Volúmenes Recomendados
```yaml
volumes:
  - ./data:/workspace/data
  - ./models:/workspace/models
  - ./logs:/workspace/logs
  - ./model-kimianet:/workspace/model-kimianet
```

## 🔬 **Aplicaciones**

### Casos de Uso Principales
- **Diagnóstico Asistido**: Mejora de imágenes de baja calidad
- **Análisis Automatizado**: Preparación para algoritmos de detección
- **Investigación**: Análisis de estructuras celulares en alta resolución
- **Archivo Digital**: Mejora de imágenes históricas de baja resolución

### Dominio Específico
- **Tipo de Imágenes**: Histopatología de cáncer de mama
- **Magnificaciones**: 10×, 20×, 40× (compatibles con escalas entrenadas)


## 📖 **Documentación Adicional**

- **[Guía Detallada de Uso](ESRGAN/README.md)**: Documentación específica del código
- **[Configuración Avanzada](ESRGAN/config/)**: Parámetros detallados
- **[Scripts de Evaluación](ESRGAN/scripts/)**: Herramientas de análisis

## 🤝 **Contribución**

Este proyecto es parte de un Trabajo de Grado. Las contribuciones son bienvenidas:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crea un Pull Request

## 📄 **Licencia**

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 **Reconocimientos**

- **Proyecto Original**: [captain-pool/GSOC E2_ESRGAN](https://github.com/captain-pool/GSOC/tree/master/E2_ESRGAN)
- **Paper ESRGAN**: Wang, Xintao, et al. "ESRGAN: Enhanced super-resolution generative adversarial networks." ECCV 2018.
- **KimiaNet**: Riasatian, Amirali, et al. "Fine-tuning and training of densenet for histopathology image representation using tcga diagnostic slides." Medical image analysis 70 (2021): 102032.
- **TensorFlow Team**: Por el framework y soporte de GPU

## 📞 **Contacto**

- **Autor**: Juan David Cruz Useche
- **Proyecto**: Trabajo de Grado - Super-Resolución para Histopatología
- **GitHub**: [@JuanHoKKeR](https://github.com/JuanHoKKeR)

---