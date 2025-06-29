#!/usr/bin/env python3
"""
Benchmark de Tiempo de Inferencia ESRGAN - VERSIÓN REALISTA
Usa imágenes reales del dataset de validación para medición precisa
"""

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import psutil
import platform
from pathlib import Path
from tqdm import tqdm
import json
import random
import warnings
warnings.filterwarnings('ignore')

class RealisticTimingBenchmark:
    """Benchmark de timing usando dataset real de validación"""
    
    def __init__(self, device='auto'):
        """
        Inicializa el benchmark
        
        Args:
            device: 'cpu', 'gpu', o 'auto'
        """
        self.device = device
        self.model = None
        self.device_info = self._get_device_info()
        self._configure_device()
        
    def _get_device_info(self):
        """Obtiene información del hardware"""
        device_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': platform.python_version(),
            'tensorflow_version': tf.__version__
        }
        
        # Información de GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            device_info['gpu_available'] = True
            device_info['gpu_count'] = len(gpus)
            try:
                gpu_details = []
                for i, gpu in enumerate(gpus):
                    gpu_details.append(f"GPU_{i}: {gpu.name}")
                device_info['gpu_details'] = gpu_details
            except:
                device_info['gpu_details'] = "GPU info not available"
        else:
            device_info['gpu_available'] = False
            device_info['gpu_count'] = 0
        
        return device_info
    
    def _configure_device(self):
        """Configura el dispositivo para el benchmark"""
        if self.device.lower() == 'cpu':
            # Forzar uso de CPU
            tf.config.set_visible_devices([], 'GPU')
            print("🖥️  Configurado para usar CPU")
            
        elif self.device.lower() == 'gpu':
            # Verificar que GPU esté disponible
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                print("⚠️  GPU solicitada pero no disponible, usando CPU")
                self.device = 'cpu'
            else:
                # Configurar memoria dinámica en GPU
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception as e:
                        print(f"⚠️  Error configurando GPU: {e}")
                print(f"🚀 Configurado para usar GPU ({len(gpus)} disponibles)")
                
        else:  # auto
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.device = 'gpu'
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except:
                        pass
                print(f"🚀 Auto-detectado: usando GPU ({len(gpus)} disponibles)")
            else:
                self.device = 'cpu'
                print("🖥️  Auto-detectado: usando CPU")
    
    def load_model(self, model_path):
        """
        Carga el modelo ESRGAN
        
        Args:
            model_path: Ruta al modelo SavedModel
        """
        print(f"📦 Cargando modelo desde: {model_path}")
        
        try:
            self.model = tf.saved_model.load(model_path)
            print("✅ Modelo cargado correctamente")
            
            # Verificar que funciona con una imagen de prueba
            test_input = tf.random.normal([1, 64, 64, 3])
            test_output = self.model(test_input)
            print(f"🧪 Prueba de inferencia exitosa - Input: {test_input.shape}, Output: {test_output.shape}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def load_validation_images(self, lr_meta_file, base_path="", max_images=50):
        """
        Carga imágenes reales del dataset de validación
        
        Args:
            lr_meta_file: Archivo meta con rutas de imágenes LR
            base_path: Ruta base para las imágenes
            max_images: Máximo número de imágenes a cargar
            
        Returns:
            Lista de tensores de imágenes cargadas
        """
        print(f"📂 Cargando imágenes reales desde: {lr_meta_file}")
        
        # Cargar rutas LR
        with open(lr_meta_file, 'r') as f:
            lr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
        
        # Limitar número de imágenes
        if len(lr_paths) > max_images:
            lr_paths = random.sample(lr_paths, max_images)
            print(f"🎲 Muestreadas {max_images} imágenes de {len(lr_paths)} disponibles")
        
        print(f"📊 Cargando {len(lr_paths)} imágenes...")
        
        images = []
        loaded_count = 0
        
        for lr_path in tqdm(lr_paths, desc="Cargando imágenes"):
            try:
                # Cargar imagen
                image = tf.io.read_file(lr_path)
                image = tf.image.decode_image(image, channels=3)
                image = tf.cast(image, tf.float32)
                
                # Verificar que la imagen se cargó correctamente
                if image.shape[0] > 0 and image.shape[1] > 0:
                    images.append(image)
                    loaded_count += 1
                else:
                    print(f"⚠️  Imagen vacía saltada: {lr_path}")
                    
            except Exception as e:
                print(f"❌ Error cargando {lr_path}: {e}")
                continue
        
        print(f"✅ Cargadas exitosamente {loaded_count} imágenes")
        
        if loaded_count == 0:
            raise ValueError("No se pudieron cargar imágenes del dataset")
        
        return images
    
    def prepare_batch_images(self, images, batch_size=1, target_size=None):
        """
        Prepara imágenes en batches del tamaño requerido
        
        Args:
            images: Lista de imágenes cargadas
            batch_size: Tamaño del batch
            target_size: Tamaño objetivo (height, width) o None para usar tamaño original
            
        Returns:
            Lista de batches preparados
        """
        batches = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # Si el batch no está completo, rellenar con imágenes repetidas
            while len(batch_images) < batch_size:
                batch_images.extend(batch_images[:batch_size-len(batch_images)])
            
            # Redimensionar si es necesario
            if target_size:
                batch_images = [
                    tf.image.resize(img, target_size, method='bicubic') 
                    for img in batch_images
                ]
            
            # Crear batch tensor
            batch_tensor = tf.stack(batch_images[:batch_size])
            batches.append(batch_tensor)
        
        return batches
    
    def measure_inference_time_realistic(self, images, batch_size=1, num_warmup=5, num_runs=20, target_size=None):
        """
        Mide el tiempo de inferencia usando imágenes reales
        
        Args:
            images: Lista de imágenes del dataset de validación
            batch_size: Tamaño del batch
            num_warmup: Número de ejecuciones de calentamiento
            num_runs: Número de ejecuciones para medir tiempo
            target_size: Tamaño objetivo (height, width) o None
            
        Returns:
            Dict con estadísticas de timing
        """
        print(f"⏱️  Midiendo tiempo de inferencia con imágenes reales...")
        print(f"   Imágenes disponibles: {len(images)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Warmup runs: {num_warmup}")
        print(f"   Timing runs: {num_runs}")
        
        # Preparar batches
        batches = self.prepare_batch_images(images, batch_size, target_size)
        
        if len(batches) == 0:
            raise ValueError("No se pudieron crear batches de imágenes")
        
        print(f"   Batches preparados: {len(batches)}")
        
        # Medir memoria antes de empezar
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # Warmup runs
        print("🔥 Ejecutando warmup...")
        for i in range(num_warmup):
            batch = batches[i % len(batches)]
            _ = self.model(batch)
        
        # Medir tiempos
        print("⏲️  Midiendo tiempos...")
        times = []
        
        for i in tqdm(range(num_runs), desc="Timing runs"):
            batch = batches[i % len(batches)]
            
            start_time = time.perf_counter()
            _ = self.model(batch)
            end_time = time.perf_counter()
            
            inference_time = end_time - start_time
            times.append(inference_time)
        
        # Medir memoria después
        memory_after = process.memory_info().rss / (1024**2)  # MB
        
        # Calcular estadísticas
        times = np.array(times)
        
        # Obtener información de la imagen
        sample_batch = batches[0]
        input_shape = sample_batch.shape
        
        stats = {
            'mean_time_ms': float(np.mean(times) * 1000),
            'std_time_ms': float(np.std(times) * 1000),
            'min_time_ms': float(np.min(times) * 1000),
            'max_time_ms': float(np.max(times) * 1000),
            'median_time_ms': float(np.median(times) * 1000),
            'p95_time_ms': float(np.percentile(times, 95) * 1000),
            'p99_time_ms': float(np.percentile(times, 99) * 1000),
            'fps': float(batch_size / np.mean(times)),
            'memory_increase_mb': float(memory_after - memory_before),
            'num_runs': num_runs,
            'batch_size': batch_size,
            'input_height': int(input_shape[1]),
            'input_width': int(input_shape[2]),
            'num_images_used': len(images),
            'num_batches_used': len(batches),
            'using_real_images': True
        }
        
        return stats
    
    def benchmark_model_realistic(self, model_path, model_name, lr_meta_file, 
                                base_path="", configurations=None):
        """
        Ejecuta benchmark completo usando dataset real de validación
        
        Args:
            model_path: Ruta al modelo
            model_name: Nombre del modelo
            lr_meta_file: Archivo meta con rutas de imágenes LR
            base_path: Ruta base para imágenes
            configurations: Lista de configuraciones a probar
        
        Returns:
            Lista de resultados
        """
        print(f"\n🏁 BENCHMARKING REALISTA - MODELO: {model_name}")
        print("=" * 60)
        
        # Cargar modelo
        self.load_model(model_path)
        
        # Cargar imágenes reales del dataset
        images = self.load_validation_images(lr_meta_file, base_path, max_images=100)
        
        # Configuraciones por defecto si no se proporcionan
        if configurations is None:
            configurations = [
                {'batch_size': 1, 'num_warmup': 5, 'num_runs': 20},
            ]
        
        results = []
        
        for i, config in enumerate(configurations):
            print(f"\n📊 Configuración {i+1}/{len(configurations)}")
            
            try:
                timing_stats = self.measure_inference_time_realistic(
                    images=images,
                    batch_size=config.get('batch_size', 1),
                    num_warmup=config.get('num_warmup', 5),
                    num_runs=config.get('num_runs', 20),
                    target_size=config.get('target_size', None)
                )
                
                # Agregar información del contexto
                result = {
                    'model_name': model_name,
                    'device': self.device,
                    'timestamp': time.time(),
                    'lr_meta_file': lr_meta_file,
                    **timing_stats,
                    **self.device_info
                }
                
                results.append(result)
                
                print(f"✅ Tiempo promedio: {timing_stats['mean_time_ms']:.2f} ms")
                print(f"   FPS: {timing_stats['fps']:.2f}")
                print(f"   Memoria adicional: {timing_stats['memory_increase_mb']:.1f} MB")
                print(f"   Imágenes reales usadas: {timing_stats['num_images_used']}")
                
            except Exception as e:
                print(f"❌ Error en configuración {i+1}: {e}")
                
        return results
    
    def save_results(self, results, output_dir, filename_prefix):
        """
        Guarda los resultados en CSV y JSON
        
        Args:
            results: Lista de resultados del benchmark
            output_dir: Directorio de salida
            filename_prefix: Prefijo para los archivos
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not results:
            print("⚠️  No hay resultados para guardar")
            return
        
        # Crear DataFrame
        df = pd.DataFrame(results)
        
        # Guardar CSV
        csv_file = os.path.join(output_dir, f"{filename_prefix}_{self.device}_timing.csv")
        df.to_csv(csv_file, index=False)
        print(f"💾 Resultados CSV guardados: {csv_file}")
        
        # Guardar JSON con más detalles
        json_file = os.path.join(output_dir, f"{filename_prefix}_{self.device}_timing.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Resultados JSON guardados: {json_file}")
        
        # Crear resumen estadístico
        self._create_timing_summary(df, output_dir, filename_prefix)
    
    def _create_timing_summary(self, df, output_dir, filename_prefix):
        """Crea un resumen estadístico de los tiempos"""
        summary_file = os.path.join(output_dir, f"{filename_prefix}_{self.device}_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"RESUMEN DE TIMING REALISTA - {self.device.upper()}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("INFORMACIÓN DEL SISTEMA:\n")
            f.write(f"Dispositivo: {self.device.upper()}\n")
            f.write(f"Platform: {self.device_info['platform']}\n")
            f.write(f"CPU: {self.device_info['processor']}\n")
            f.write(f"Cores: {self.device_info['cpu_count']}\n")
            f.write(f"RAM: {self.device_info['ram_gb']} GB\n")
            if self.device_info['gpu_available']:
                f.write(f"GPU: Disponible ({self.device_info['gpu_count']} unidades)\n")
            else:
                f.write("GPU: No disponible\n")
            f.write(f"TensorFlow: {self.device_info['tensorflow_version']}\n\n")
            
            f.write("ESTADÍSTICAS DE TIMING CON IMÁGENES REALES:\n")
            for model in df['model_name'].unique():
                model_data = df[df['model_name'] == model]
                f.write(f"\n{model}:\n")
                f.write(f"  Tiempo promedio: {model_data['mean_time_ms'].mean():.2f} ms\n")
                f.write(f"  FPS promedio: {model_data['fps'].mean():.2f}\n")
                f.write(f"  Memoria promedio: {model_data['memory_increase_mb'].mean():.1f} MB\n")
                f.write(f"  Imágenes reales usadas: {model_data['num_images_used'].iloc[0]}\n")
                f.write(f"  Tamaño imagen: {model_data['input_height'].iloc[0]}x{model_data['input_width'].iloc[0]}\n")
        
        print(f"📄 Resumen guardado: {summary_file}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Benchmark Realista de Tiempo de Inferencia ESRGAN")
    
    parser.add_argument(
        "--model_path",
        required=True,
        help="Ruta al modelo ESRGAN (directorio SavedModel)"
    )
    
    parser.add_argument(
        "--model_name",
        required=True,
        help="Nombre del modelo (ej: 128to512)"
    )
    
    parser.add_argument(
        "--lr_meta_file",
        required=True,
        help="Archivo meta con rutas de imágenes LR del dataset de validación"
    )
    
    parser.add_argument(
        "--base_path",
        default="",
        help="Ruta base para las imágenes"
    )
    
    parser.add_argument(
        "--device",
        choices=['cpu', 'gpu', 'auto'],
        default='auto',
        help="Dispositivo a usar para benchmark"
    )
    
    parser.add_argument(
        "--batch_sizes",
        nargs='+',
        type=int,
        default=[1],
        help="Tamaños de batch a probar (ej: 1 2 4)"
    )
    
    parser.add_argument(
        "--num_runs",
        type=int,
        default=20,
        help="Número de ejecuciones para timing"
    )
    
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=5,
        help="Número de ejecuciones de warmup"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=50,
        help="Máximo número de imágenes del dataset a usar"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./realistic_timing_results",
        help="Directorio para guardar resultados"
    )
    
    args = parser.parse_args()
    
    # Verificar que los archivos existen
    required_files = [args.model_path, args.lr_meta_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Error: No se encuentra: {file_path}")
            return 1
    
    print("⚡ BENCHMARK REALISTA DE TIEMPO DE INFERENCIA ESRGAN")
    print("=" * 60)
    print(f"Modelo: {args.model_name}")
    print(f"Ruta: {args.model_path}")
    print(f"Dataset LR: {args.lr_meta_file}")
    print(f"Dispositivo: {args.device}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Runs: {args.num_runs}")
    print(f"Imágenes máx: {args.max_images}")
    
    try:
        # Inicializar benchmark
        benchmark = RealisticTimingBenchmark(device=args.device)
        
        # Crear configuraciones de prueba
        configurations = []
        for batch_size in args.batch_sizes:
            configurations.append({
                'batch_size': batch_size,
                'num_runs': args.num_runs,
                'num_warmup': args.num_warmup
            })
        
        # Ejecutar benchmark realista
        results = benchmark.benchmark_model_realistic(
            args.model_path,
            args.model_name,
            args.lr_meta_file,
            args.base_path,
            configurations
        )
        
        # Guardar resultados
        benchmark.save_results(results, args.output_dir, args.model_name)
        
        print(f"\n🎉 Benchmark realista completado exitosamente!")
        print(f"📂 Resultados guardados en: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Error durante el benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())