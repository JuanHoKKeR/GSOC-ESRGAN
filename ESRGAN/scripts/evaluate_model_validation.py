#!/usr/bin/env python3
"""
Evaluador de Validación para Modelo ESRGAN Específico
Evalúa un modelo específico con su dataset de validación correspondiente
Genera CSVs con métricas en color y blanco y negro
"""

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from pathlib import Path
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ESRGANModelEvaluator:
    """Evaluador para un modelo ESRGAN específico"""
    
    def __init__(self, kimianet_weights_path):
        """
        Inicializa el evaluador
        
        Args:
            kimianet_weights_path: Ruta a los pesos de KimiaNet
        """
        self.kimianet_weights_path = kimianet_weights_path
        self.model = None
        self.feature_extractor = None
        self._initialize_kimianet()
        
    def _initialize_kimianet(self):
        """Inicializa DenseNet121 con pesos KimiaNet para índice perceptual"""
        print("🧠 Inicializando KimiaNet para índice perceptual...")
        
        self.densenet = DenseNet121(
            include_top=False, 
            weights=None,
            input_shape=(None, None, 3)
        )
        
        if os.path.exists(self.kimianet_weights_path):
            try:
                self.densenet.load_weights(self.kimianet_weights_path)
                print(f"✅ Pesos KimiaNet cargados desde: {self.kimianet_weights_path}")
            except Exception as e:
                print(f"⚠️  Error cargando pesos KimiaNet: {e}")
                print("    Continuando sin pesos preentrenados")
        else:
            print(f"⚠️  No se encontró el archivo de pesos: {self.kimianet_weights_path}")
        
        # Usar capa intermedia para características
        try:
            feature_layer = self.densenet.get_layer('conv4_block6_concat')
        except:
            try:
                feature_layer = self.densenet.get_layer('conv4_block24_concat')
            except:
                feature_layer = self.densenet.layers[-2]
        
        self.feature_extractor = tf.keras.Model(
            inputs=self.densenet.input,
            outputs=feature_layer.output
        )
        
        for layer in self.feature_extractor.layers:
            layer.trainable = False
            
        print(f"✅ Extractor de características listo: {feature_layer.name}")
    
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
    
    def load_validation_dataset(self, lr_meta_file, hr_meta_file, base_path=""):
        """
        Carga el dataset de validación desde archivos paired_meta
        
        Args:
            lr_meta_file: Archivo con rutas de imágenes LR
            hr_meta_file: Archivo con rutas de imágenes HR  
            base_path: Ruta base para las imágenes
            
        Returns:
            Lista de tuplas (lr_path, hr_path)
        """
        print(f"📂 Cargando dataset de validación...")
        print(f"   LR meta: {lr_meta_file}")
        print(f"   HR meta: {hr_meta_file}")
        
        # Cargar rutas LR
        with open(lr_meta_file, 'r') as f:
            lr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
        
        # Cargar rutas HR
        with open(hr_meta_file, 'r') as f:
            hr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
        
        print(f"✅ Cargadas {len(lr_paths)} imágenes LR y {len(hr_paths)} imágenes HR")
        
        # Verificar que las cantidades coinciden
        if len(lr_paths) != len(hr_paths):
            print(f"⚠️  Advertencia: Número diferente de imágenes LR ({len(lr_paths)}) y HR ({len(hr_paths)})")
        
        # Verificar que existen algunas imágenes de muestra
        sample_size = min(5, len(lr_paths))
        missing_files = 0
        for i in range(sample_size):
            if not os.path.exists(lr_paths[i]):
                missing_files += 1
                print(f"⚠️  Archivo LR no encontrado: {lr_paths[i]}")
            if not os.path.exists(hr_paths[i]):
                missing_files += 1
                print(f"⚠️  Archivo HR no encontrado: {hr_paths[i]}")
        
        if missing_files > 0:
            print(f"⚠️  Se encontraron {missing_files} archivos faltantes en la muestra")
        
        # Crear pares de rutas
        image_pairs = list(zip(lr_paths, hr_paths))
        print(f"📊 Dataset preparado con {len(image_pairs)} pares de imágenes")
        
        return image_pairs
    
    def load_image(self, image_path):
        """Carga una imagen y la convierte a tensor float32"""
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.cast(image, tf.float32)
            return image
        except Exception as e:
            print(f"❌ Error cargando imagen {image_path}: {e}")
            return None
    
    def convert_to_grayscale(self, image):
        """Convierte imagen a escala de grises - VERSIÓN CORREGIDA"""
        try:
            # SOLUCIÓN: Conversión en CPU para evitar MatMul en GPU
            with tf.device('/CPU:0'):
                # Conversión manual usando pesos estándar
                r, g, b = tf.split(image, 3, axis=-1)
                grayscale = 0.299 * r + 0.587 * g + 0.114 * b
                # Convertir de vuelta a 3 canales
                grayscale_3ch = tf.concat([grayscale, grayscale, grayscale], axis=-1)
                return grayscale_3ch
        except Exception as e:
            print(f"⚠️  Error conversión a grises: {e}")
            # Fallback: promedio simple
            gray_simple = tf.reduce_mean(image, axis=-1, keepdims=True)
            return tf.concat([gray_simple, gray_simple, gray_simple], axis=-1)
    
    def calculate_perceptual_index(self, img1, img2):
        """Calcula índice perceptual usando KimiaNet"""
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.cast(img2, tf.float32)
        
        # Agregar dimensión de batch si no existe
        if len(img1.shape) == 3:
            img1 = tf.expand_dims(img1, 0)
        if len(img2.shape) == 3:
            img2 = tf.expand_dims(img2, 0)
        
        # Normalización para DenseNet
        img1_norm = (img1 - 127.5) / 127.5
        img2_norm = (img2 - 127.5) / 127.5
        
        # Extraer características
        features1 = self.feature_extractor(img1_norm)
        features2 = self.feature_extractor(img2_norm)
        
        # Distancia L2 entre características
        perceptual_distance = tf.reduce_mean(tf.square(features1 - features2))
        
        return perceptual_distance
    
    def calculate_all_metrics(self, generated, hr_image):
        """
        Calcula todas las métricas de evaluación
        
        Args:
            generated: Imagen generada por ESRGAN
            hr_image: Imagen de alta resolución (ground truth)
            
        Returns:
            Dict con todas las métricas
        """
        # Asegurar que las imágenes tengan el mismo tamaño
        if generated.shape != hr_image.shape:
            hr_image = tf.image.resize(hr_image, generated.shape[:2], method='bicubic')
        
        # Métricas básicas
        psnr = tf.image.psnr(generated, hr_image, max_val=255.0)
        ssim = tf.image.ssim(generated, hr_image, max_val=255.0)
        
        # MS-SSIM (con manejo de errores para imágenes pequeñas)
        try:
            ms_ssim = tf.image.ssim_multiscale(
                tf.expand_dims(generated, 0), 
                tf.expand_dims(hr_image, 0), 
                max_val=255.0
            )
            ms_ssim = tf.squeeze(ms_ssim)
        except:
            ms_ssim = ssim  # Usar SSIM regular si MS-SSIM falla
        
        # MSE
        mse = tf.reduce_mean(tf.square(generated - hr_image))
        
        # Índice perceptual
        perceptual_index = self.calculate_perceptual_index(generated, hr_image)
        
        return {
            'psnr': float(psnr.numpy()),
            'ssim': float(ssim.numpy()),
            'ms_ssim': float(ms_ssim.numpy()),
            'mse': float(mse.numpy()),
            'perceptual_index': float(perceptual_index.numpy())
        }
    
    def evaluate_single_pair(self, lr_path, hr_path):
        """
        Evalúa un par de imágenes LR-HR
        
        Args:
            lr_path: Ruta a imagen de baja resolución
            hr_path: Ruta a imagen de alta resolución
            
        Returns:
            Dict con métricas en color y blanco y negro, o None si hay error
        """
        try:
            # Cargar imágenes
            lr_image = self.load_image(lr_path)
            hr_image = self.load_image(hr_path)
            
            if lr_image is None or hr_image is None:
                return None
            
            # Generar imagen con ESRGAN
            lr_batch = tf.expand_dims(lr_image, 0)
            generated_batch = self.model(lr_batch)
            generated = tf.squeeze(generated_batch, 0)
            generated = tf.clip_by_value(generated, 0, 255)
            
            # Métricas en COLOR
            color_metrics = self.calculate_all_metrics(generated, hr_image)
            
            # Convertir a escala de grises
            generated_gray = self.convert_to_grayscale(generated)
            hr_image_gray = self.convert_to_grayscale(hr_image)
            
            # Métricas en BLANCO Y NEGRO
            gray_metrics = self.calculate_all_metrics(generated_gray, hr_image_gray)
            
            return {
                'lr_path': lr_path,
                'hr_path': hr_path,
                'color_metrics': color_metrics,
                'gray_metrics': gray_metrics
            }
            
        except Exception as e:
            print(f"❌ Error procesando par {lr_path} - {hr_path}: {e}")
            return None
    
    def evaluate_model(self, image_pairs, output_dir, model_name):
        """
        Evalúa el modelo con todos los pares de imágenes
        
        Args:
            image_pairs: Lista de tuplas (lr_path, hr_path)
            output_dir: Directorio para guardar resultados
            model_name: Nombre del modelo para los archivos de salida
        """
        print(f"\n🚀 Iniciando evaluación del modelo {model_name}")
        print(f"📊 Procesando {len(image_pairs)} pares de imágenes...")
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Listas para almacenar resultados
        color_results = []
        gray_results = []
        
        # Procesar cada par de imágenes
        successful_evaluations = 0
        failed_evaluations = 0
        
        for i, (lr_path, hr_path) in enumerate(tqdm(image_pairs, desc="Evaluando imágenes")):
            result = self.evaluate_single_pair(lr_path, hr_path)
            
            if result is not None:
                # Preparar datos para CSV
                base_data = {
                    'image_index': i + 1,
                    'lr_path': result['lr_path'],
                    'hr_path': result['hr_path'],
                    'lr_filename': os.path.basename(result['lr_path']),
                    'hr_filename': os.path.basename(result['hr_path'])
                }
                
                # Datos para CSV de color
                color_row = base_data.copy()
                color_row.update(result['color_metrics'])
                color_results.append(color_row)
                
                # Datos para CSV de escala de grises
                gray_row = base_data.copy()
                gray_row.update(result['gray_metrics'])
                gray_results.append(gray_row)
                
                successful_evaluations += 1
            else:
                failed_evaluations += 1
        
        print(f"\n📈 Evaluación completada:")
        print(f"   ✅ Exitosas: {successful_evaluations}")
        print(f"   ❌ Fallidas: {failed_evaluations}")
        
        # Crear DataFrames y guardar CSVs
        if color_results:
            # CSV para métricas en color
            color_df = pd.DataFrame(color_results)
            color_csv_path = os.path.join(output_dir, f"{model_name}_metrics_color.csv")
            color_df.to_csv(color_csv_path, index=False)
            print(f"💾 Métricas en COLOR guardadas: {color_csv_path}")
            
            # CSV para métricas en escala de grises
            gray_df = pd.DataFrame(gray_results)
            gray_csv_path = os.path.join(output_dir, f"{model_name}_metrics_grayscale.csv")
            gray_df.to_csv(gray_csv_path, index=False)
            print(f"💾 Métricas en ESCALA DE GRISES guardadas: {gray_csv_path}")
            
            # Mostrar estadísticas resumidas
            self._print_summary_statistics(color_df, gray_df, model_name)
            
        else:
            print("❌ No se pudieron evaluar imágenes correctamente")
    
    def _print_summary_statistics(self, color_df, gray_df, model_name):
        """Imprime estadísticas resumidas"""
        print(f"\n📊 ESTADÍSTICAS RESUMIDAS - {model_name}")
        print("=" * 50)
        
        metrics = ['psnr', 'ssim', 'ms_ssim', 'mse', 'perceptual_index']
        
        print("COLOR:")
        for metric in metrics:
            if metric in color_df.columns:
                mean_val = color_df[metric].mean()
                std_val = color_df[metric].std()
                print(f"  {metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")
        
        print("\nESCALA DE GRISES:")
        for metric in metrics:
            if metric in gray_df.columns:
                mean_val = gray_df[metric].mean()
                std_val = gray_df[metric].std()
                print(f"  {metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Evaluador de Validación para Modelo ESRGAN Específico")
    
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
        help="Archivo meta con rutas de imágenes LR (ej: paired_lr_meta.txt)"
    )
    
    parser.add_argument(
        "--hr_meta_file",
        required=True,
        help="Archivo meta con rutas de imágenes HR (ej: paired_hr_meta.txt)"
    )
    
    parser.add_argument(
        "--kimianet_weights",
        default="/workspace/model-kimianet/KimiaNetKerasWeights.h5",
        help="Ruta a los pesos de KimiaNet"
    )
    
    parser.add_argument(
        "--base_path",
        default="",
        help="Ruta base para las imágenes (si las rutas en meta_info son relativas)"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./evaluation_results",
        help="Directorio para guardar los resultados CSV"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Máximo número de imágenes a evaluar (para pruebas rápidas)"
    )
    
    args = parser.parse_args()
    
    # Verificar que los archivos existen
    required_files = [args.model_path, args.lr_meta_file, args.hr_meta_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Error: No se encuentra el archivo/directorio: {file_path}")
            return 1
    
    print("🔬 EVALUADOR DE MODELO ESRGAN")
    print("=" * 40)
    print(f"Modelo: {args.model_name}")
    print(f"Ruta del modelo: {args.model_path}")
    print(f"Dataset LR: {args.lr_meta_file}")
    print(f"Dataset HR: {args.hr_meta_file}")
    print(f"KimiaNet: {args.kimianet_weights}")
    print(f"Resultados: {args.output_dir}")
    
    try:
        # Inicializar evaluador
        evaluator = ESRGANModelEvaluator(args.kimianet_weights)
        
        # Cargar modelo
        evaluator.load_model(args.model_path)
        
        # Cargar dataset de validación
        image_pairs = evaluator.load_validation_dataset(
            args.lr_meta_file, args.hr_meta_file, args.base_path
        )
        
        # Limitar número de imágenes si se especifica
        if args.max_images and args.max_images < len(image_pairs):
            image_pairs = image_pairs[:args.max_images]
            print(f"🔢 Limitando evaluación a {args.max_images} imágenes")
        
        # Evaluar modelo
        evaluator.evaluate_model(image_pairs, args.output_dir, args.model_name)
        
        print(f"\n🎉 Evaluación completada exitosamente!")
        print(f"📂 Archivos CSV generados en: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Error fatal durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())