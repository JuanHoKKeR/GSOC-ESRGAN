#!/usr/bin/env python3
"""
Super-Resolución Basada en Parches ESRGAN
Divide imagen en parches, aplica modelo, reconstruye imagen completa
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class PatchBasedSuperResolution:
    """Aplicación de super-resolución por parches"""
    
    def __init__(self, model_path, patch_size=512):
        """
        Inicializa el sistema de super-resolución por parches
        
        Args:
            model_path: Ruta al modelo ESRGAN
            patch_size: Tamaño de los parches (debe coincidir con entrada del modelo)
        """
        self.patch_size = patch_size
        self.model = None
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Carga el modelo ESRGAN"""
        print(f"📦 Cargando modelo desde: {model_path}")
        
        try:
            self.model = tf.saved_model.load(model_path)
            print("✅ Modelo cargado correctamente")
            
            # Verificar funcionamiento y determinar escala
            test_input = tf.random.normal([1, self.patch_size, self.patch_size, 3])
            test_output = self.model(test_input)
            
            self.scale_factor = test_output.shape[1] // test_input.shape[1]
            self.output_patch_size = test_output.shape[1]
            
            print(f"🔍 Modelo detectado:")
            print(f"   Input: {test_input.shape[1]}x{test_input.shape[2]} → Output: {test_output.shape[1]}x{test_output.shape[2]}")
            print(f"   Factor de escala: ×{self.scale_factor}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def load_image(self, image_path):
        """Carga una imagen y la convierte a tensor float32"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"No se encuentra el archivo: {image_path}")
                
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.cast(image, tf.float32)
            
            # Verificar que la imagen se cargó correctamente
            if image.shape.rank != 3 or image.shape[-1] != 3:
                raise ValueError(f"Imagen inválida. Esperado: [H,W,3], obtenido: {image.shape}")
                
            return image
        except Exception as e:
            print(f"❌ Error cargando imagen {image_path}: {e}")
            return None
    
    def extract_patches(self, image, patch_size):
        """
        Extrae parches de una imagen
        
        Args:
            image: Imagen tensor [H, W, C]
            patch_size: Tamaño del parche
            
        Returns:
            Lista de parches y información de grilla
        """
        h, w = image.shape[:2]
        
        # Calcular número de parches
        patches_h = h // patch_size
        patches_w = w // patch_size
        
        print(f"📐 Dividiendo imagen {h}x{w} en {patches_h}x{patches_w} parches de {patch_size}x{patch_size}")
        
        if h % patch_size != 0 or w % patch_size != 0:
            print(f"⚠️  Advertencia: La imagen no se divide exactamente en parches de {patch_size}x{patch_size}")
            print(f"    Se usarán solo los parches completos")
        
        patches = []
        patch_positions = []
        
        for i in range(patches_h):
            for j in range(patches_w):
                y_start = i * patch_size
                y_end = y_start + patch_size
                x_start = j * patch_size
                x_end = x_start + patch_size
                
                patch = image[y_start:y_end, x_start:x_end, :]
                patches.append(patch)
                patch_positions.append((i, j, y_start, y_end, x_start, x_end))
        
        grid_info = {
            'patches_h': patches_h,
            'patches_w': patches_w,
            'patch_size': patch_size,
            'original_size': (h, w),
            'positions': patch_positions
        }
        
        print(f"✅ Extraídos {len(patches)} parches")
        return patches, grid_info
    
    def process_patches(self, patches):
        """
        Procesa cada parche con el modelo ESRGAN
        
        Args:
            patches: Lista de parches tensor
            
        Returns:
            Lista de parches procesados (super-resolución)
        """
        processed_patches = []
        
        print(f"🚀 Procesando {len(patches)} parches con ESRGAN...")
        
        for i, patch in enumerate(patches):
            print(f"   Procesando parche {i+1}/{len(patches)}", end='\r')
            
            # Agregar dimensión de batch
            patch_batch = tf.expand_dims(patch, 0)
            
            # Aplicar modelo
            enhanced_batch = self.model(patch_batch)
            enhanced_patch = tf.squeeze(enhanced_batch, 0)
            enhanced_patch = tf.clip_by_value(enhanced_patch, 0, 255)
            
            processed_patches.append(enhanced_patch)
        
        print(f"\n✅ Todos los parches procesados")
        return processed_patches
    
    def reconstruct_image(self, processed_patches, grid_info):
        """
        Reconstruye la imagen completa a partir de parches procesados
        
        Args:
            processed_patches: Lista de parches super-resueltos
            grid_info: Información de la grilla original
            
        Returns:
            Imagen reconstruida
        """
        patches_h = grid_info['patches_h']
        patches_w = grid_info['patches_w']
        original_h, original_w = grid_info['original_size']
        
        # Calcular tamaño de la imagen reconstruida
        reconstructed_h = original_h * self.scale_factor
        reconstructed_w = original_w * self.scale_factor
        
        print(f"🔧 Reconstruyendo imagen: {original_h}x{original_w} → {reconstructed_h}x{reconstructed_w}")
        
        # Inicializar imagen reconstruida
        reconstructed = tf.zeros([reconstructed_h, reconstructed_w, 3], dtype=tf.float32)
        
        # Colocar cada parche en su posición
        patch_idx = 0
        for i in range(patches_h):
            for j in range(patches_w):
                # Calcular posición en la imagen reconstruida
                y_start = i * self.output_patch_size
                y_end = y_start + self.output_patch_size
                x_start = j * self.output_patch_size
                x_end = x_start + self.output_patch_size
                
                # Colocar parche
                patch = processed_patches[patch_idx]
                
                # Usar tf.tensor_scatter_nd_update para actualizar la imagen
                indices = []
                updates = []
                
                for y in range(self.output_patch_size):
                    for x in range(self.output_patch_size):
                        for c in range(3):
                            indices.append([y_start + y, x_start + x, c])
                            updates.append(patch[y, x, c])
                
                indices = tf.constant(indices)
                updates = tf.constant(updates)
                reconstructed = tf.tensor_scatter_nd_update(reconstructed, indices, updates)
                
                patch_idx += 1
        
        print(f"✅ Imagen reconstruida: {reconstructed.shape}")
        return reconstructed
    
    def calculate_metrics(self, generated, ground_truth):
        """Calcula métricas de evaluación"""
        # Asegurar mismo tamaño
        if generated.shape != ground_truth.shape:
            print(f"⚠️  Redimensionando ground truth: {ground_truth.shape} → {generated.shape}")
            ground_truth = tf.image.resize(ground_truth, generated.shape[:2], method='bicubic')
        
        # Calcular métricas
        psnr = tf.image.psnr(generated, ground_truth, max_val=255.0)
        ssim = tf.image.ssim(generated, ground_truth, max_val=255.0)
        
        # MS-SSIM con manejo de errores
        try:
            ms_ssim = tf.image.ssim_multiscale(
                tf.expand_dims(generated, 0), 
                tf.expand_dims(ground_truth, 0), 
                max_val=255.0
            )
            ms_ssim = tf.squeeze(ms_ssim)
        except:
            ms_ssim = ssim
        
        # MSE
        mse = tf.reduce_mean(tf.square(generated - ground_truth))
        
        return {
            'psnr': float(psnr.numpy()),
            'ssim': float(ssim.numpy()),
            'ms_ssim': float(ms_ssim.numpy()),
            'mse': float(mse.numpy())
        }
    
    def calculate_absolute_difference(self, generated, ground_truth):
        """Calcula diferencia absoluta"""
        # Normalizar a [0, 1]
        gen_norm = generated / 255.0
        gt_norm = ground_truth / 255.0
        
        # Diferencia absoluta
        abs_diff = tf.abs(gen_norm - gt_norm)
        
        return abs_diff
    
    def save_results(self, generated_image, ground_truth, abs_diff, metrics, output_dir, image_name):
        """Guarda todos los resultados"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar imagen generada
        generated_path = os.path.join(output_dir, f"{image_name}_patch_reconstructed.png")
        generated_pil = Image.fromarray(tf.cast(tf.clip_by_value(generated_image, 0, 255), tf.uint8).numpy())
        generated_pil.save(generated_path)
        print(f"💾 Imagen reconstruida guardada: {generated_path}")

        # Guardar diferencia absoluta
        diff_path = os.path.join(output_dir, f"{image_name}_absolute_difference.png")

        # Crear visualización de diferencia
        plt.figure(figsize=(12, 5))
        
        # Imagen reconstruida
        plt.subplot(1, 3, 1)
        plt.imshow(generated_pil)
        plt.title('Patch-based Reconstruction', fontweight='bold')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(1, 3, 2)
        gt_pil = Image.fromarray(tf.cast(tf.clip_by_value(ground_truth, 0, 255), tf.uint8).numpy())
        plt.imshow(gt_pil)
        plt.title('Ground Truth', fontweight='bold')
        plt.axis('off')
        
        # Diferencia absoluta
        plt.subplot(1, 3, 3)
        diff_display = np.mean(abs_diff.numpy(), axis=-1)  # Promedio de canales para visualización
        im = plt.imshow(diff_display, cmap='hot', vmin=0, vmax=0.3)
        plt.title(f'Absolute Difference\nMean Error: {np.mean(diff_display):.4f}', fontweight='bold')
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        # Título general con métricas
        plt.suptitle(f'Patch-based Super-Resolution Results - {image_name}\n' +
                    f'PSNR: {metrics["psnr"]:.4f} dB | SSIM: {metrics["ssim"]:.4f} | MSE: {metrics["mse"]:.2f}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(diff_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📊 Análisis de diferencia guardado: {diff_path}")

        # Guardar métricas en JSON
        metrics_path = os.path.join(output_dir, f"{image_name}_patch_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'patch_info': {
                    'model_scale_factor': self.scale_factor,
                    'patch_size': self.patch_size,
                    'output_patch_size': self.output_patch_size,
                    'generated_size': generated_image.shape[:2].as_list(),
                    'ground_truth_size': ground_truth.shape[:2].as_list()
                }
            }, f, indent=2)
        print(f"📄 Métricas guardadas: {metrics_path}")

    def process_image_pair(self, input_1024_path, ground_truth_2048_path, output_dir, image_name=None):
        """
        Procesa un par de imágenes: 1024 → parches → reconstrucción 2048
        
        Args:
            input_1024_path: Imagen de entrada 1024x1024
            ground_truth_2048_path: Imagen ground truth 2048x2048
            output_dir: Directorio de salida
            image_name: Nombre para archivos de salida
        """
        if image_name is None:
            image_name = Path(input_1024_path).stem

        print(f"\n🎯 PROCESANDO: {image_name}")
        print("=" * 50)
        
        # Cargar imágenes
        print("📂 Cargando imágenes...")
        input_1024 = self.load_image(input_1024_path)
        ground_truth_2048 = self.load_image(ground_truth_2048_path)
        
        if input_1024 is None or ground_truth_2048 is None:
            print("❌ Error cargando imágenes")
            return None

        print(f"   Input 1024: {input_1024.shape}")
        print(f"   Ground Truth 2048: {ground_truth_2048.shape}")

        # Verificar tamaños
        if input_1024.shape[0] != 1024 or input_1024.shape[1] != 1024:
            print(f"⚠️  Advertencia: Imagen de entrada no es 1024x1024, redimensionando...")
            input_1024 = tf.image.resize(input_1024, [1024, 1024], method='bicubic')
        
        # Extraer parches
        patches, grid_info = self.extract_patches(input_1024, self.patch_size)
        
        # Procesar parches
        processed_patches = self.process_patches(patches)
        
        # Reconstruir imagen
        reconstructed_2048 = self.reconstruct_image(processed_patches, grid_info)
        
        # Calcular métricas
        print("📊 Calculando métricas...")
        metrics = self.calculate_metrics(reconstructed_2048, ground_truth_2048)
        
        # Calcular diferencia absoluta
        abs_diff = self.calculate_absolute_difference(reconstructed_2048, ground_truth_2048)
        
        # Mostrar resultados
        print(f"\n📈 RESULTADOS:")
        print(f"   PSNR: {metrics['psnr']:.4f} dB")
        print(f"   SSIM: {metrics['ssim']:.4f}")
        print(f"   MS-SSIM: {metrics['ms_ssim']:.4f}")
        print(f"   MSE: {metrics['mse']:.2f}")

        # Guardar resultados
        self.save_results(reconstructed_2048, ground_truth_2048, abs_diff, metrics, output_dir, image_name)

        print(f"✅ Procesamiento completado para {image_name}")
        return metrics

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Super-Resolución Basada en Parches ESRGAN")
    
    parser.add_argument(
        "--model_path",
        required=True,
        help="Ruta al modelo ESRGAN (ej: modelo 512→1024)"
    )
    
    parser.add_argument(
        "--input_1024",
        required=True,
        help="Imagen de entrada 1024x1024"
    )
    
    parser.add_argument(
        "--ground_truth_2048",
        required=True,
        help="Imagen ground truth 2048x2048"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./patch_based_results",
        help="Directorio para guardar resultados"
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Tamaño de parches (debe coincidir con entrada del modelo)"
    )
    
    parser.add_argument(
        "--image_name",
        default=None,
        help="Nombre personalizado para archivos de salida"
    )
    
    args = parser.parse_args()
    
    # Verificar archivos
    for file_path in [args.model_path, args.input_1024, args.ground_truth_2048]:
        if not os.path.exists(file_path):
            print(f"❌ Error: No se encuentra: {file_path}")
            return 1

    print("🧩 SUPER-RESOLUCIÓN BASADA EN PARCHES")
    print("=" * 50)
    print(f"Modelo: {args.model_path}")
    print(f"Input 1024: {args.input_1024}")
    print(f"Ground Truth 2048: {args.ground_truth_2048}")
    print(f"Tamaño de parche: {args.patch_size}x{args.patch_size}")
    print(f"Output: {args.output_dir}")

    try:
        # Crear procesador
        processor = PatchBasedSuperResolution(args.model_path, args.patch_size)
        
        # Procesar imágenes
        metrics = processor.process_image_pair(
            args.input_1024,
            args.ground_truth_2048,
            args.output_dir,
            args.image_name
        )
        
        if metrics:
            print(f"\n🎉 Procesamiento exitoso!")
            print(f"📂 Resultados guardados en: {args.output_dir}")

        return 0
        
    except Exception as e:
        print(f"\n💥 Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())