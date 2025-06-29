#!/usr/bin/env python3
"""
Analizador Visual de Diferencias para ESRGAN
Crea visualizaciones comprehensivas de diferencias entre predicción y ground truth
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import tensorflow as tf
from pathlib import Path
import json
from skimage import filters, feature, segmentation
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class ESRGANDifferenceAnalyzer:
    """Analizador visual de diferencias entre predicción y ground truth"""
    
    def __init__(self):
        """Inicializa el analizador"""
        self.model = None
        
    def load_model(self, model_path):
        """Carga el modelo ESRGAN"""
        print(f"📦 Cargando modelo desde: {model_path}")
        
        try:
            self.model = tf.saved_model.load(model_path)
            print("✅ Modelo cargado correctamente")
            
            # Verificar funcionamiento
            test_input = tf.random.normal([1, 64, 64, 3])
            test_output = self.model(test_input)
            print(f"🧪 Prueba exitosa - Input: {test_input.shape}, Output: {test_output.shape}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
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
    
    def normalize_images(self, pred_image, gt_image, range_type='0_1'):
        """
        Normaliza las imágenes al mismo rango
        
        Args:
            pred_image: Imagen predicha
            gt_image: Imagen ground truth
            range_type: '0_1' o '0_255'
        """
        if range_type == '0_1':
            pred_norm = pred_image / 255.0
            gt_norm = gt_image / 255.0
        else:  # '0_255'
            pred_norm = tf.clip_by_value(pred_image, 0, 255)
            gt_norm = tf.clip_by_value(gt_image, 0, 255)
        
        return pred_norm, gt_norm
    
    def convert_to_grayscale(self, image):
        """Convierte imagen a escala de grises - VERSIÓN CORREGIDA"""
        try:
            # SOLUCIÓN: Conversión en CPU para evitar MatMul en GPU
            with tf.device('/CPU:0'):
                # Conversión manual usando pesos estándar
                r, g, b = tf.split(image, 3, axis=-1)
                grayscale = 0.299 * r + 0.587 * g + 0.114 * b
                return tf.squeeze(grayscale, axis=-1)  # Retornar como 2D para diferencias
        except Exception as e:
            print(f"⚠️  Error conversión a grises: {e}")
            # Fallback: promedio simple
            gray_simple = tf.reduce_mean(image, axis=-1)
            return gray_simple

    def calculate_differences(self, pred_image, gt_image):
        """
        Calcula múltiples tipos de diferencias
        
        Args:
            pred_image: Imagen predicha
            gt_image: Imagen ground truth
            
        Returns:
            Dict con diferentes tipos de diferencias
        """
        # Asegurar mismo tamaño
        if pred_image.shape != gt_image.shape:
            gt_image = tf.image.resize(gt_image, pred_image.shape[:2], method='bicubic')
        
        # Normalizar a [0, 1]
        pred_norm, gt_norm = self.normalize_images(pred_image, gt_image, '0_1')
        
        # Diferencia absoluta
        abs_diff = tf.abs(pred_norm - gt_norm)
        
        # Diferencia cuadrática
        squared_diff = tf.square(pred_norm - gt_norm)
        
        # Diferencia firmada (conserva signo)
        signed_diff = pred_norm - gt_norm
        
        # Diferencia por canal
        abs_diff_r = abs_diff[:, :, 0]
        abs_diff_g = abs_diff[:, :, 1] 
        abs_diff_b = abs_diff[:, :, 2]
        
        # Diferencia en escala de grises - VERSIÓN CORREGIDA
        pred_gray = self.convert_to_grayscale(pred_norm)
        gt_gray = self.convert_to_grayscale(gt_norm)
        gray_diff = tf.abs(pred_gray - gt_gray)
        
        return {
            'absolute': abs_diff.numpy(),
            'squared': squared_diff.numpy(),
            'signed': signed_diff.numpy(),
            'channel_r': abs_diff_r.numpy(),
            'channel_g': abs_diff_g.numpy(),
            'channel_b': abs_diff_b.numpy(),
            'grayscale': gray_diff.numpy(),
            'pred_normalized': pred_norm.numpy(),
            'gt_normalized': gt_norm.numpy()
        }
    
    def analyze_error_regions(self, abs_diff, threshold=0.1):
        """
        Analiza regiones con diferentes niveles de error
        
        Args:
            abs_diff: Diferencia absoluta
            threshold: Umbral para considerar "alto error"
            
        Returns:
            Dict con análisis de regiones
        """
        # Promedio de diferencia por píxel
        pixel_error = np.mean(abs_diff, axis=-1)
        
        # Crear máscaras de error
        low_error_mask = pixel_error < threshold/2
        medium_error_mask = (pixel_error >= threshold/2) & (pixel_error < threshold)
        high_error_mask = pixel_error >= threshold
        
        # Estadísticas por región
        total_pixels = pixel_error.size
        
        analysis = {
            'low_error_percentage': np.sum(low_error_mask) / total_pixels * 100,
            'medium_error_percentage': np.sum(medium_error_mask) / total_pixels * 100,
            'high_error_percentage': np.sum(high_error_mask) / total_pixels * 100,
            'mean_error': np.mean(pixel_error),
            'max_error': np.max(pixel_error),
            'std_error': np.std(pixel_error),
            'error_map': pixel_error,
            'masks': {
                'low': low_error_mask,
                'medium': medium_error_mask,
                'high': high_error_mask
            }
        }
        
        return analysis
    
    def detect_edges_and_textures(self, image):
        """
        Detecta bordes y regiones con textura
    
        Args:
            image: Imagen normalizada [0, 1]
        
        Returns:
            Dict con máscaras de bordes y texturas
        """
        # Convertir a escala de grises para análisis
        if len(image.shape) == 3:
            gray = np.mean(image, axis=-1)
        else:
            gray = image
    
        # Detectar bordes con Canny
        edges = feature.canny(gray, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    
        # Detectar texturas usando filtro de varianza local - VERSIÓN CORREGIDA
        try:
            # Intentar con el módulo correcto
            from skimage.filters.rank import variance
            texture_mask = variance(
                (gray * 255).astype(np.uint8), 
                np.ones((5, 5))
            ) > 500
        except ImportError:
            try:
                # Alternativa con filtros genéricos
                from skimage.filters import rank
                texture_mask = rank.variance(
                    (gray * 255).astype(np.uint8), 
                    np.ones((5, 5))
                ) > 500
            except (ImportError, AttributeError):
                # Fallback: usar desviación estándar local con scipy
                print("⚠️  Usando filtro de textura alternativo")
                gray_uint8 = (gray * 255).astype(np.uint8)
                local_std = ndimage.generic_filter(
                    gray_uint8.astype(np.float32), 
                    np.std, 
                    size=5
                )
                texture_mask = local_std > 20  # Umbral ajustado para std
    
        # Regiones lisas (ni bordes ni texturas)
        smooth_mask = ~(edges | texture_mask)
    
        return {
            'edges': edges,
            'textures': texture_mask,
            'smooth': smooth_mask
        }
    
    def create_comprehensive_visualization(self, lr_image, pred_image, gt_image, 
                                         output_path, image_name="image"):
        """
        Crea visualización comprehensiva de diferencias
        
        Args:
            lr_image: Imagen de baja resolución
            pred_image: Imagen predicha por ESRGAN
            gt_image: Imagen ground truth
            output_path: Ruta para guardar la visualización
            image_name: Nombre de la imagen para títulos
        """
        # Calcular diferencias
        differences = self.calculate_differences(pred_image, gt_image)
        
        # Análisis de regiones de error
        error_analysis = self.analyze_error_regions(differences['absolute'])
        
        # Análisis de bordes y texturas en ground truth
        content_analysis = self.detect_edges_and_textures(differences['gt_normalized'])
        
        # Crear figura grande con múltiples subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.3)
        
        # Redimensionar LR para visualización
        lr_resized = tf.image.resize(lr_image, pred_image.shape[:2], method='nearest')
        lr_resized = lr_resized.numpy() / 255.0
        
        # Fila 1: Imágenes principales
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(lr_resized)
        ax1.set_title(f'LR Input\n{lr_image.shape[0]}x{lr_image.shape[1]}', fontsize=10, pad=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(differences['pred_normalized'])
        ax2.set_title(f'ESRGAN Prediction\n{pred_image.shape[0]}x{pred_image.shape[1]}', fontsize=10, pad=10)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(differences['gt_normalized'])
        ax3.set_title(f'Ground Truth\n{gt_image.shape[0]}x{gt_image.shape[1]}', fontsize=10, pad=10)
        ax3.axis('off')
        
        # Diferencia absoluta total
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(np.mean(differences['absolute'], axis=-1), cmap='hot', vmin=0, vmax=0.3)
        ax4.set_title(f'Absolute Difference\nMean: {error_analysis["mean_error"]:.4f}', fontsize=10, pad=10)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        # Mapa de error categorizado
        ax5 = fig.add_subplot(gs[0, 4])
        error_colored = np.zeros((*error_analysis['error_map'].shape, 3))
        error_colored[error_analysis['masks']['low']] = [0, 1, 0]      # Verde: bajo error
        error_colored[error_analysis['masks']['medium']] = [1, 1, 0]  # Amarillo: medio error
        error_colored[error_analysis['masks']['high']] = [1, 0, 0]    # Rojo: alto error
        ax5.imshow(error_colored)
        ax5.set_title('Error Regions\nGreen: Low | Yellow: Med | Red: High', fontsize=10, pad=10)
        ax5.axis('off')
        
        # Fila 2: Diferencias por canal
        channels = ['R', 'G', 'B']
        channel_diffs = [differences['channel_r'], differences['channel_g'], differences['channel_b']]
        
        for i, (channel, diff) in enumerate(zip(channels, channel_diffs)):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(diff, cmap='Reds', vmin=0, vmax=0.3)
            ax.set_title(f'Channel {channel} Diff\nMean: {np.mean(diff):.4f}', fontsize=10, pad=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Diferencia en escala de grises
        ax_gray = fig.add_subplot(gs[1, 3])
        im_gray = ax_gray.imshow(differences['grayscale'], cmap='gray', vmin=0, vmax=0.3)
        ax_gray.set_title(f'Grayscale Diff\nMean: {np.mean(differences["grayscale"]):.4f}', fontsize=10, pad=10)
        ax_gray.axis('off')
        plt.colorbar(im_gray, ax=ax_gray, fraction=0.046, pad=0.04)
        
        # Diferencia cuadrática
        ax_sq = fig.add_subplot(gs[1, 4])
        im_sq = ax_sq.imshow(np.mean(differences['squared'], axis=-1), cmap='plasma', vmin=0, vmax=0.1)
        ax_sq.set_title(f'Squared Difference\nMean: {np.mean(differences["squared"]):.4f}', fontsize=10, pad=10)
        ax_sq.axis('off')
        plt.colorbar(im_sq, ax=ax_sq, fraction=0.046, pad=0.04)
        
        # Fila 3: Análisis de contenido
        ax_edges = fig.add_subplot(gs[2, 0])
        ax_edges.imshow(content_analysis['edges'], cmap='gray')
        ax_edges.set_title('Detected Edges', fontsize=10, pad=10)
        ax_edges.axis('off')
        
        ax_textures = fig.add_subplot(gs[2, 1])
        ax_textures.imshow(content_analysis['textures'], cmap='gray')
        ax_textures.set_title('Texture Regions', fontsize=10, pad=10)
        ax_textures.axis('off')
        
        ax_smooth = fig.add_subplot(gs[2, 2])
        ax_smooth.imshow(content_analysis['smooth'], cmap='gray')
        ax_smooth.set_title('Smooth Regions', fontsize=10, pad=10)
        ax_smooth.axis('off')
        
        # Error en bordes vs regiones lisas
        edge_errors = error_analysis['error_map'][content_analysis['edges']]
        smooth_errors = error_analysis['error_map'][content_analysis['smooth']]
        texture_errors = error_analysis['error_map'][content_analysis['textures']]
        
        ax_edge_err = fig.add_subplot(gs[2, 3])
        edge_error_mask = np.zeros_like(error_analysis['error_map'])
        edge_error_mask[content_analysis['edges']] = error_analysis['error_map'][content_analysis['edges']]
        im_edge = ax_edge_err.imshow(edge_error_mask, cmap='Reds', vmin=0, vmax=0.3)
        ax_edge_err.set_title(f'Error at Edges\nMean: {np.mean(edge_errors):.4f}', fontsize=10, pad=10)
        ax_edge_err.axis('off')
        plt.colorbar(im_edge, ax=ax_edge_err, fraction=0.046, pad=0.04)
        
        # Histograma de errores
        ax_hist = fig.add_subplot(gs[2, 4])
        ax_hist.hist(error_analysis['error_map'].flatten(), bins=50, alpha=0.7, color='blue', label='All pixels')
        if len(edge_errors) > 0:
            ax_hist.hist(edge_errors, bins=30, alpha=0.7, color='red', label='Edges')
        if len(smooth_errors) > 0:
            ax_hist.hist(smooth_errors, bins=30, alpha=0.7, color='green', label='Smooth')
        ax_hist.set_xlabel('Error Magnitude')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Error Distribution')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Fila 4: Estadísticas y métricas
        ax_stats = fig.add_subplot(gs[3, :3])
        ax_stats.axis('off')
        
        # Crear tabla de estadísticas
        stats_text = f"""
        ERROR ANALYSIS - {image_name}
        {'='*50}
        
        Overall Statistics:
        • Mean Error: {error_analysis['mean_error']:.4f}
        • Max Error: {error_analysis['max_error']:.4f}
        • Std Error: {error_analysis['std_error']:.4f}
        
        Error Distribution:
        • Low Error (< 5%): {error_analysis['low_error_percentage']:.1f}% of pixels
        • Medium Error (5-10%): {error_analysis['medium_error_percentage']:.1f}% of pixels  
        • High Error (> 10%): {error_analysis['high_error_percentage']:.1f}% of pixels
        
        Content-based Analysis:
        • Edge Pixels: {np.sum(content_analysis['edges'])} ({np.sum(content_analysis['edges'])/content_analysis['edges'].size*100:.1f}%)
        • Texture Pixels: {np.sum(content_analysis['textures'])} ({np.sum(content_analysis['textures'])/content_analysis['textures'].size*100:.1f}%)
        • Smooth Pixels: {np.sum(content_analysis['smooth'])} ({np.sum(content_analysis['smooth'])/content_analysis['smooth'].size*100:.1f}%)
        
        Error by Content Type:
        • Mean Error at Edges: {np.mean(edge_errors) if len(edge_errors) > 0 else 0:.4f}
        • Mean Error in Textures: {np.mean(texture_errors) if len(texture_errors) > 0 else 0:.4f}
        • Mean Error in Smooth Areas: {np.mean(smooth_errors) if len(smooth_errors) > 0 else 0:.4f}
        """
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                     verticalalignment='top', fontfamily='monospace', fontsize=9)
        
        # Gráfico de barras de errores por tipo de contenido
        ax_bar = fig.add_subplot(gs[3, 3:])
        content_types = ['Edges', 'Textures', 'Smooth']
        mean_errors = [
            np.mean(edge_errors) if len(edge_errors) > 0 else 0,
            np.mean(texture_errors) if len(texture_errors) > 0 else 0,
            np.mean(smooth_errors) if len(smooth_errors) > 0 else 0
        ]
        
        bars = ax_bar.bar(content_types, mean_errors, color=['red', 'orange', 'green'], alpha=0.7)
        ax_bar.set_ylabel('Mean Error')
        ax_bar.set_title('Mean Error by Content Type')
        ax_bar.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bar, value in zip(bars, mean_errors):
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom')
        
        # Título general
        fig.suptitle(f'Comprehensive Difference Analysis - {image_name}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Guardar figura
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Guardar estadísticas en JSON
        stats_output = output_path.replace('.png', '_stats.json')
        
        # Convertir todos los valores NumPy a tipos nativos de Python
        error_analysis_json = {}
        for k, v in error_analysis.items():
            if k != 'masks':  # Excluir máscaras que son arrays
                if isinstance(v, np.ndarray):
                    if v.size == 1:  # Si es un escalar
                        error_analysis_json[k] = float(v.item())
                    else:  # Si es un array, convertir a lista
                        error_analysis_json[k] = v.tolist()
                else:
                    error_analysis_json[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        
        with open(stats_output, 'w') as f:
            json.dump({
                'error_analysis': error_analysis_json,
                'content_analysis': {
                    'edge_percentage': float(np.sum(content_analysis['edges'])/content_analysis['edges'].size*100),
                    'texture_percentage': float(np.sum(content_analysis['textures'])/content_analysis['textures'].size*100),
                    'smooth_percentage': float(np.sum(content_analysis['smooth'])/content_analysis['smooth'].size*100)
                },
                'error_by_content': {
                    'edges': float(np.mean(edge_errors)) if len(edge_errors) > 0 else 0.0,
                    'textures': float(np.mean(texture_errors)) if len(texture_errors) > 0 else 0.0,
                    'smooth': float(np.mean(smooth_errors)) if len(smooth_errors) > 0 else 0.0
                }
            }, f, indent=2)
        
        print(f"✅ Visualización guardada: {output_path}")
        print(f"📊 Estadísticas guardadas: {stats_output}")
        
        return error_analysis, content_analysis
    
    def analyze_image_pair(self, lr_path, hr_path, output_dir, model_path=None):
        """
        Analiza un par de imágenes LR-HR específico
        
        Args:
            lr_path: Ruta a imagen de baja resolución
            hr_path: Ruta a imagen de alta resolución
            output_dir: Directorio para guardar resultados
            model_path: Ruta al modelo (si no está ya cargado)
        """
        if model_path and not self.model:
            self.load_model(model_path)
        
        if not self.model:
            raise ValueError("Modelo no cargado. Proporciona model_path o carga el modelo primero.")
        
        # Cargar imágenes
        lr_image = self.load_image(lr_path)
        hr_image = self.load_image(hr_path)
        
        if lr_image is None or hr_image is None:
            raise ValueError("Error cargando imágenes")
        
        # Generar predicción
        lr_batch = tf.expand_dims(lr_image, 0)
        pred_batch = self.model(lr_batch)
        pred_image = tf.squeeze(pred_batch, 0)
        pred_image = tf.clip_by_value(pred_image, 0, 255)
        
        # Crear nombre de archivo de salida
        lr_name = Path(lr_path).stem
        output_path = os.path.join(output_dir, f"{lr_name}_difference_analysis.png")
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear visualización
        error_analysis, content_analysis = self.create_comprehensive_visualization(
            lr_image, pred_image, hr_image, output_path, lr_name
        )
        
        return error_analysis, content_analysis

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Analizador Visual de Diferencias ESRGAN")
    
    parser.add_argument(
        "--lr_image",
        required=True,
        help="Ruta a imagen de baja resolución"
    )
    
    parser.add_argument(
        "--hr_image", 
        required=True,
        help="Ruta a imagen de alta resolución (ground truth)"
    )
    
    parser.add_argument(
        "--model_path",
        required=True,
        help="Ruta al modelo ESRGAN"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./difference_analysis",
        help="Directorio para guardar análisis visual"
    )
    
    args = parser.parse_args()
    
    # Verificar archivos
    for file_path in [args.lr_image, args.hr_image, args.model_path]:
        if not os.path.exists(file_path):
            print(f"❌ Error: No se encuentra: {file_path}")
            return 1
    
    print("🎨 ANALIZADOR VISUAL DE DIFERENCIAS ESRGAN")
    print("=" * 50)
    print(f"LR Image: {args.lr_image}")
    print(f"HR Image: {args.hr_image}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    
    try:
        # Crear analizador
        analyzer = ESRGANDifferenceAnalyzer()
        
        # Analizar par de imágenes
        error_analysis, content_analysis = analyzer.analyze_image_pair(
            args.lr_image, args.hr_image, args.output_dir, args.model_path
        )
        
        print(f"\n🎉 Análisis visual completado!")
        print(f"📂 Resultados guardados en: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())