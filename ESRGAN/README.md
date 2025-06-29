# ESRGAN Implementation for Histopathology

Esta carpeta contiene la implementación específica de ESRGAN optimizada para imágenes de histopatología. Incluye scripts de entrenamiento, evaluación, modelos preentrenados y herramientas de análisis.

## 📁 **Estructura del Proyecto**

```
ESRGAN/
├── config/                     # Archivos de configuración
│   ├── config.yaml            # Configuración principal de entrenamiento
│   └── stats.yaml             # Estado del entrenamiento
├── datasets/                  # Metadatos de datasets
│   ├── paired_meta/           # Archivos meta pareados LR-HR
│   └── validation/            # Datasets de validación por modelo
├── Exported_Models/           # Modelos entrenados exportados
│   ├── ESRGAN_128to256_*/     # Modelo 128→256
│   ├── ESRGAN_256to512_*/     # Modelo 256→512
│   └── ...                    # Otros modelos entrenados
├── lib/                       # Biblioteca principal
│   ├── dataset.py            # Manejo de datasets y carga de datos
│   ├── model.py              # Arquitecturas (RRDBNet, VGGArch, DenseNet)
│   ├── settings.py           # Configuración y parámetros
│   ├── train.py              # Lógica de entrenamiento
│   └── utils.py              # Utilidades y funciones auxiliares
├── scripts/                   # Scripts de evaluación y análisis
│   ├── evaluate_model.py     # Evaluación individual de modelos
│   ├── compare_sr.py         # Comparación entre modelos
│   ├── export_trained_model.py  # Exportación de modelos
│   └── ...                   # Otros scripts de utilidad
├── train_microscopy_meta_info.py  # Script principal de entrenamiento
├── train_microscopy.sh       # Script de bash para entrenamiento
└── requirements.txt          # Dependencias específicas
```

## 🚀 **Scripts Principales**

### 1. **Entrenamiento de Modelos**

#### `train_microscopy_meta_info.py`
Script principal para entrenar modelos ESRGAN con datasets de histopatología.

```bash
python train_microscopy_meta_info.py \
  --hr_meta_file 'datasets/paired_meta/paired_hr_meta.txt' \
  --lr_meta_file 'datasets/paired_meta/paired_lr_meta.txt' \
  --model_dir '/workspace/Exported_Models/ESRGAN_64to128_KimiaNet' \
  --log_dir '/workspace/logs' \
  --config 'config/config.yaml' \
  --wandb_project 'ESRGAN-Microscopy-X2' \
  --wandb_name 'Microscopy_64to128_KimiaNet' \
  --phase 'phase1_phase2' \
  -v
```

**Parámetros importantes:**
- `--hr_meta_file`: Archivo con rutas de imágenes de alta resolución
- `--lr_meta_file`: Archivo con rutas de imágenes de baja resolución  
- `--model_dir`: Directorio para guardar checkpoints del modelo
- `--log_dir`: Directorio para logs de TensorBoard
- `--config`: Archivo de configuración YAML
- `--kimianet_weights`: Ruta a pesos preentrenados de KimiaNet
- `--phase`: Fase de entrenamiento (`phase1`, `phase2`, `phase1_phase2`)
- `--pretrained_model`: Modelo preentrenado para fine-tuning
- `--v`: Verbosidad del proceso de entrenamiento

#### Fases de Entrenamiento

**Fase 1 - Warmup (PSNR)**:
- Entrena solo el generador con pérdida L1
- Optimiza métricas pixel-wise (PSNR)
- Prepara el generador para entrenamiento GAN

**Fase 2 - GAN Training**:
- Entrenamiento adversarial completo
- Incluye pérdida perceptual y adversarial
- Utiliza discriminador DenseNet+KimiaNet

### 2. **Evaluación de Modelos**

#### `scripts/use_savedmodelTest.py`
Genera una imagen apartir de un modelo dado y una imagen de LR de entrada.

```bash
python scripts/use_savedmodelTest.py \
    --model Exported_Models/ESRGAN_512to1024_KimiaNet/esrgan \
    --input LR_512x512.jpg \
    --output Predict_HR_1024x1024.jpg
```

#### Ejemplo de Resultado Visual

![Resultado visual 512→1024 KimiaNet](ESRGAN/Validation/Visual_Evaluations/512to1024_KimiaNet.jpg)

#### `scripts/evaluate_model.py`
Evaluación individual con métricas completas.

```bash
python scripts/evaluate_model.py \
    --generated path/to/generated_image.jpg \
    --ground_truth path/to/original_image.jpg \
    --kimianet_weights ../model-kimianet/KimiaNetKerasWeights.h5
```

**Métricas calculadas:**
- PSNR (Peak Signal-to-Noise Ratio)
- Índice Perceptual (basado en KimiaNet)

#### Evaluación con Dataset de Validación
```bash
# Script personalizado para evaluación comprehensive
python scripts/evaluate_model_validation.py \
    --model_path Exported_Models/ESRGAN_256to512/esrgan \
    --model_name "256to512_KimiaNet" \
    --lr_meta_file datasets/validation/256to512/paired_lr_meta.txt \
    --hr_meta_file datasets/validation/256to512/paired_hr_meta.txt \
    --kimianet_weights ../model-kimianet/KimiaNetKerasWeights.h5 \
    --output_dir ./evaluation_results \
    --max_images 1000
```

**Genera:**
- `modelo_metrics_color.csv`: Métricas para imágenes en color
- `modelo_metrics_grayscale.csv`: Métricas para escala de grises
- Estadísticas resumidas en consola

### 3. **Análisis Visual**

#### Análisis de Diferencias
```bash
python scripts/visual_difference_analyzer.py \
    --lr_image path/to/low_res.jpg \
    --hr_image path/to/high_res.jpg \
    --model_path Exported_Models/ESRGAN_256to512/esrgan \
    --output_dir ./visual_analysis
```

**Genera análisis comprehensive:**
- Comparación lado a lado (LR → Predicción → GT)
- Mapas de diferencia por canal RGB
- Análisis de errores por tipo de contenido (bordes, texturas, áreas lisas)
- Estadísticas detalladas en JSON

#### Ejemplo de Análisis de Diferencias

![Análisis de diferencias - Patch 734](ESRGAN/Validation/Visual_Evaluations/TCGA-VQ-AA6B-01Z-00-DX1.C6CB7290-4A83-4963-8840-68BC208D1232_40x_512px_x23040_y20992_Patch734_difference_analysis.png)

### 4. **Benchmarking de Rendimiento**

#### Timing de Inferencia
```bash
python scripts/inference_timing_benchmark.py \
    --model_path Exported_Models/ESRGAN_256to512/esrgan \
    --model_name "256to512" \
    --device gpu \
    --batch_sizes 1 2 4 \
    --num_runs 50 \
    --output_dir ./timing_results
```

**Análisis de rendimiento:**
- Tiempo de inferencia (GPU/CPU)
- Throughput (FPS)
- Uso de memoria
- Comparación entre dispositivos

### 5. **Super-Resolución por Parches**

#### Escalado Extremo
```bash
python scripts/patch_based_superresolution.py \
    --model_path Exported_Models/ESRGAN_512to1024/esrgan \
    --input_1024 path/to/image_1024.png \
    --ground_truth_2048 path/to/image_2048.png \
    --patch_size 512 \
    --output_dir ./patch_results
```

**Para escalas no disponibles directamente:**
- Divide imagen en parches
- Aplica modelo a cada parche
- Reconstruye imagen completa
- Evalúa resultado vs ground truth

#### Ejemplo de Mapa de Diferencias Absolutas

![Mapa de diferencias absolutas](ESRGAN/Validation/test_image_patch_v1_absolute_difference.png)

## ⚙️ **Configuración Detallada**

### `config/config.yaml`

#### Configuración de Dataset
```yaml
dataset:
  hr_dimension: 256        # Resolución objetivo (potencias de 2)
  lr_dimension: 128        # Resolución de entrada  
  name: "Microscopy"       # Nombre del dataset
  scale_method: bicubic    # Método de escalado (bicubic, bilinear, nearest)
```

#### Configuración del Generador
```yaml
generator:
  num_features: 32         # Características base (16, 32, 64)
  trunk_size: 11          # Número de bloques RRDB (6, 11, 23)
  growth_channel: 32      # Canales de crecimiento en RDB (16, 32, 64)
```

**Configuraciones probadas:**
- **Estándar**: `(32, 11, 32)` - Mejor balance calidad/memoria
- **Optimizado memoria**: `(16, 6, 16)` - Para 512→1024
- **Alto rendimiento**: `(64, 23, 64)` - Mayor calidad, más memoria

#### Configuración del Discriminador
```yaml
discriminator:
  type: densenet                    # densenet | vgg | optimized_vgg
  kimianet_weights: /path/to/KimiaNetKerasWeights.h5
  num_features: 64                  # Solo para VGG (32, 64)
```

**Tipos de discriminador:**
- **`densenet`**: DenseNet121 + KimiaNet (recomendado para histopatología)
- **`vgg`**: VGG28 estándar (compatible con ESRGAN original)
- **`optimized_vgg`**: VGG optimizado para memoria limitada

#### Hiperparámetros de Entrenamiento
```yaml
# Fase 1 - PSNR Training
train_psnr:
  num_steps: 60000
  adam:
    initial_lr: 0.0003
    beta_1: 0.9
    beta_2: 0.999
    decay:
      step: 30000
      factor: 0.5

# Fase 2 - GAN Training  
train_combined:
  num_steps: 50000
  lambda: 0.005              # Peso pérdida adversarial
  eta: 0.01                 # Peso pérdida L1
  perceptual_loss_type: L1  # L1 | L2
  adam:
    initial_lr: 2.0e-05
    discriminator_lr: 5.0e-06
    decay:
      step: [2000, 6000, 10000, 15000, 25000]
      factor: 0.5
```

## 🧠 **Arquitecturas de Modelos**

### Generador (RRDBNet)
```python
# En lib/model.py
RRDBNet(
    out_channel=3,
    num_features=32,        # Configurable
    trunk_size=11,          # Configurable  
    growth_channel=32,      # Configurable
    upscale_factor=2        # Calculado automáticamente
)
```

### Discriminadores Disponibles

#### DenseNet + KimiaNet (Recomendado)
```python
DenseNetDiscriminator(
    output_shape=1,
    kimianet_weights_path="path/to/weights.h5"
)
```
- **Ventajas**: Especializado en histopatología, mejor discriminación
- **Desventajas**: Requiere pesos preentrenados

#### VGG28 Estándar
```python
VGGArch(
    batch_size=4,
    num_features=64,
    output_shape=1
)
```
- **Ventajas**: Compatible con ESRGAN original, no requiere pesos extra
- **Desventajas**: Menos especializado para imágenes médicas

#### VGG Optimizado
```python
OptimizedVGGArch(
    batch_size=4,
    num_features=32,    # Reducido para memoria
    output_shape=1
)
```
- **Ventajas**: Menor uso de memoria, más rápido
- **Desventajas**: Posiblemente menor calidad de discriminación

## 📊 **Sistema de Métricas**

### Métricas Implementadas

#### Pixel-wise Metrics
```python
# En lib/utils.py
psnr = tf.image.psnr(generated, ground_truth, max_val=255.0)
ssim = tf.image.ssim(generated, ground_truth, max_val=255.0)  
ms_ssim = tf.image.ssim_multiscale(generated, ground_truth, max_val=255.0)
mse = tf.reduce_mean(tf.square(generated - ground_truth))
```

#### Perceptual Metrics
```python
# Índice perceptual usando KimiaNet
perceptual_loss = KimiaNetPerceptualLoss(kimianet_weights_path)
perceptual_index = perceptual_loss.calculate_perceptual_distance(generated, ground_truth)
```

### Evaluación Dual: Color y Escala de Grises
```python
# Conversión a escala de grises optimizada
def convert_to_grayscale(self, image):
    with tf.device('/CPU:0'):  # Evitar problemas de memoria GPU
        r, g, b = tf.split(image, 3, axis=-1)
        grayscale = 0.299 * r + 0.587 * g + 0.114 * b
        return tf.concat([grayscale, grayscale, grayscale], axis=-1)
```

## 🔄 **Flujo de Trabajo Típico**

### 1. Preparación de Datos
```bash
# Organizar dataset
mkdir -p datasets/paired_meta
echo "/path/to/hr/image1.png" > datasets/paired_meta/paired_hr_meta.txt
echo "/path/to/lr/image1.png" > datasets/paired_meta/paired_lr_meta.txt
```

### 2. Configurar Modelo
```bash
# Editar config/config.yaml según memoria disponible y objetivos
# Ajustar batch_size, generator params, discriminator type
```

### 3. Entrenamiento
```bash
# Fase 1 + 2
python train_microscopy_meta_info.py \
    --hr_meta_file datasets/paired_meta/paired_hr_meta.txt \
    --lr_meta_file datasets/paired_meta/paired_lr_meta.txt \
    --model_dir ./models/experiment_1 \
    --log_dir ./logs/experiment_1 \
    --config config/config.yaml \
    --phase phase1_phase2
```

### 4. Exportar Modelo (En caso de que la Fase 2 no se complete correctamente) 
```bash
python scripts/export_trained_model.py \
    --model_dir ./models/experiment_1 \
    --output_dir ./Exported_Models/ESRGAN_custom
```

### 5. Evaluación
```bash
# Evaluar con dataset de validación
python evaluate_model_validation.py \
    --model_path ./Exported_Models/ESRGAN_custom \
    --model_name "custom_model" \
    --lr_meta_file datasets/validation/custom/paired_lr_meta.txt \
    --hr_meta_file datasets/validation/custom/paired_hr_meta.txt
```

### 6. Análisis de Resultados
```bash
# Análisis estadístico comprehensive
python results_analysis_system.py \
    --validation_dir ./Validation \
    --output_dir ./analysis_results
```

## 🐛 **Solución de Problemas Comunes**

### Problemas de Memoria GPU

**Error**: `ResourceExhaustedError: OOM when allocating tensor`

**Soluciones:**
1. **Reducir batch_size**: En `config.yaml` o como parámetro
2. **Optimizar arquitectura**: Usar parámetros más pequeños
   ```yaml
   generator:
     num_features: 16      # En lugar de 32
     trunk_size: 6         # En lugar de 11
     growth_channel: 16    # En lugar de 32
   ```
3. **Usar discriminador optimizado**:
   ```yaml
   discriminator:
     type: optimized_vgg
     num_features: 32      # En lugar de 64
   ```

### Problemas de Gradientes

**Error**: Red genera imágenes incorrectas/ruido

**Causas posibles:**
- Parámetros demasiado grandes (ej: `trunk_size: 23`)
- Learning rate muy alto
- Pérdida de gradientes en arquitecturas profundas

**Soluciones:**
1. **Usar configuración estándar**: `(32, 11, 32)`
2. **Reducir learning rate**: Dividir por 2 o 5
3. **Monitorear training**: Verificar logs de TensorBoard

### Problemas de Conversión a Escala de Grises

**Error**: GPU memory exhausted durante conversión

**Solución**: Ya implementada en el código
```python
# Forzar CPU para operaciones problemáticas
with tf.device('/CPU:0'):
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b
```

## 📈 **Monitoreo y Logs**

### TensorBoard
```bash
# Visualizar entrenamiento
tensorboard --logdir ./logs/experiment_1

# Métricas disponibles:
# - warmup_loss (Fase 1)
# - mean_psnr (Fase 1 y 2)  
# - gen_loss, disc_loss (Fase 2)
# - mean_ssim, mean_mse (ambas fases)
```

### Weights & Biases (Opcional)
```bash
# Configurar W&B
pip install wandb
wandb login

# Entrenar con tracking
python train_microscopy_meta_info.py \
    --wandb_project "mi-proyecto-esrgan" \
    --wandb_name "experimento-256to512" \
    # ... otros parámetros
```

## 🔧 **Scripts de Utilidad**

### `scripts/check_image_sizes.py`
Verifica dimensiones de imágenes en dataset.

### `scripts/prepare_meta_pairs.py`
Crea archivos meta_info pareados desde directorios.

### `scripts/monitor_memory.py`
Monitorea uso de memoria durante entrenamiento.

### `scripts/verify_tensorflow.py`
Verifica instalación y detecta GPU.

## 📚 **Referencias y Recursos**

### Papers Relacionados
- **ESRGAN Original**: Wang, Xintao, et al. "ESRGAN: Enhanced super-resolution generative adversarial networks." ECCV 2018.
- **KimiaNet**: Riasatian, Amirali, et al. "Fine-tuning and training of densenet for histopathology image representation using tcga diagnostic slides." Medical image analysis 70 (2021): 102032.

### Recursos de KimiaNet
- **Sitio web**: https://kimialab.uwaterloo.ca/kimia/
- **Paper**: https://doi.org/10.1016/j.media.2021.102032
- **Pesos preentrenados**: https://github.com/KimiaLabMayo/KimiaNet/tree/main/KimiaNet_Weights/weights

---

## 🚨 **Notas Importantes**

1. **GPU Recomendada**: RTX 3080/4090+ con 10GB+ VRAM para modelos estándar
2. **Tiempo de Entrenamiento**: 12-48 horas según configuración y dataset
3. **Espacio en Disco**: ~5-20GB por modelo entrenado (incluye checkpoints)
4. **Compatibilidad**: Optimizado para TensorFlow 2.11, CUDA 11.2+
