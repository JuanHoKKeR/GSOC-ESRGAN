#!/usr/bin/env python3
"""
Analizador de Resultados de Superresolución
Compara imagen LR, predicción, HR original y genera análisis visual completo
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configurar fuente Computer Modern Roman para matplotlib
plt.rcParams.update({
    'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'font.size': 11
})


class KimiaNetPerceptualLoss:
    """Índice perceptual usando DenseNet+KimiaNet para histopatología"""
    
    def __init__(self, kimianet_weights_path):
        print("🧠 Cargando DenseNet121 con pesos KimiaNet...")
        
        # Cargar DenseNet121 sin la capa final
        self.densenet = DenseNet121(
            include_top=False, 
            weights=None,
            input_shape=(None, None, 3)
        )
        
        # Cargar pesos KimiaNet si existe el archivo
        if kimianet_weights_path and os.path.exists(kimianet_weights_path):
            try:
                self.densenet.load_weights(kimianet_weights_path)
                print(f"✅ Pesos KimiaNet cargados desde: {kimianet_weights_path}")
            except Exception as e:
                print(f"⚠️  Error cargando pesos KimiaNet: {e}")
                print("    Usando DenseNet121 sin preentrenar")
        else:
            print("⚠️  No se encontraron pesos KimiaNet, usando DenseNet121 sin preentrenar")
        
        # Usar una capa intermedia para extraer características
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
        
        # Congelar el modelo  
        for layer in self.feature_extractor.layers:
            layer.trainable = False
            
        print(f"✅ Extractor de características listo: {feature_layer.name}")
    
    def calculate_perceptual_distance(self, img1, img2):
        """Calcula distancia perceptual entre dos imágenes usando KimiaNet"""
        # Asegurar que sean tensores float32
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.cast(img2, tf.float32)
        
        # Agregar dimensión de batch si no existe
        if len(img1.shape) == 3:
            img1 = tf.expand_dims(img1, 0)
        if len(img2.shape) == 3:
            img2 = tf.expand_dims(img2, 0)
        
        # Normalizar para DenseNet
        img1_norm = (img1 - 127.5) / 127.5  # [-1, 1]
        img2_norm = (img2 - 127.5) / 127.5  # [-1, 1]
        
        # Extraer características
        features1 = self.feature_extractor(img1_norm)
        features2 = self.feature_extractor(img2_norm)
        
        # Calcular distancia L2 entre características
        perceptual_distance = tf.reduce_mean(tf.square(features1 - features2))
        
        return perceptual_distance


class SuperResolutionAnalyzer:
    """Analizador completo de resultados de superresolución"""
    
    def __init__(self, kimianet_weights_path=None):
        """Inicializa el analizador"""
        self.perceptual_evaluator = None
        if kimianet_weights_path:
            try:
                self.perceptual_evaluator = KimiaNetPerceptualLoss(kimianet_weights_path)
            except Exception as e:
                print(f"⚠️  Error inicializando evaluador perceptual: {e}")
    
    def load_image(self, image_path, target_size=None):
        """Carga una imagen y opcionalmente la redimensiona"""
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.cast(image, tf.float32)
            
            if target_size is not None:
                image = tf.image.resize(image, target_size, method='bicubic')
            
            return image
        except Exception as e:
            print(f"❌ Error cargando {image_path}: {e}")
            return None
    
    def calculate_metrics(self, predicted_img, ground_truth_img):
        """Calcula todas las métricas de evaluación"""
        
        # Asegurar mismo tamaño
        if predicted_img.shape != ground_truth_img.shape:
            print(f"⚠️  Redimensionando ground truth: {ground_truth_img.shape} → {predicted_img.shape}")
            ground_truth_img = tf.image.resize(ground_truth_img, predicted_img.shape[:2], method='bicubic')
        
        # PSNR y SSIM
        psnr = tf.image.psnr(predicted_img, ground_truth_img, max_val=255.0)
        ssim = tf.image.ssim(predicted_img, ground_truth_img, max_val=255.0)
        
        # MS-SSIM con manejo de errores
        try:
            ms_ssim = tf.image.ssim_multiscale(
                tf.expand_dims(predicted_img, 0),
                tf.expand_dims(ground_truth_img, 0),
                max_val=255.0
            )
            ms_ssim = tf.squeeze(ms_ssim)
        except:
            ms_ssim = ssim
        
        # MSE y MAE
        mse = tf.reduce_mean(tf.square(predicted_img - ground_truth_img))
        mae = tf.reduce_mean(tf.abs(predicted_img - ground_truth_img))
        
        metrics = {
            'psnr': float(psnr.numpy()),
            'ssim': float(ssim.numpy()),
            'ms_ssim': float(ms_ssim.numpy()),
            'mse': float(mse.numpy()),
            'mae': float(mae.numpy())
        }
        
        # Índice perceptual si está disponible
        if self.perceptual_evaluator:
            try:
                perceptual_dist = self.perceptual_evaluator.calculate_perceptual_distance(
                    predicted_img, ground_truth_img
                )
                metrics['perceptual_index'] = float(perceptual_dist.numpy())
            except Exception as e:
                print(f"⚠️  Error calculando índice perceptual: {e}")
                metrics['perceptual_index'] = None
        else:
            metrics['perceptual_index'] = None
        
        return metrics
    
    def calculate_absolute_difference(self, predicted_img, ground_truth_img):
        """Calcula diferencia absoluta normalizada"""
        # Normalizar a [0, 1]
        pred_norm = predicted_img / 255.0
        gt_norm = ground_truth_img / 255.0
        
        # Diferencia absoluta
        abs_diff = tf.abs(pred_norm - gt_norm)
        
        return abs_diff
    
    def generate_latex_table(self, metrics, output_path, image_name=""):
        """Genera tabla LaTeX con las métricas"""
        
        scale_factor = "Determinado por tamaños de imagen"
        
        latex_table = f"""\\begin{{table}}[!htb]
\\centering
\\caption{{Métricas de evaluación para superresolución: {image_name}}}
\\label{{tab:sr_metrics_{image_name.lower().replace(' ', '_')}}}
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Métrica}} & \\textbf{{Valor}} \\\\
\\hline
PSNR (dB) & {metrics['psnr']:.4f} \\\\
SSIM & {metrics['ssim']:.4f} \\\\
MS-SSIM & {metrics['ms_ssim']:.4f} \\\\
MSE & {metrics['mse']:.2f} \\\\
MAE & {metrics['mae']:.2f} \\\\
"""
        
        if metrics['perceptual_index'] is not None:
            latex_table += f"Índice Perceptual & {metrics['perceptual_index']:.6f} \\\\\n"
        
        latex_table += """\\hline
\\end{tabular}
\\textbf{Notas:} PSNR más alto indica mejor fidelidad. SSIM evalúa similitud estructural (0-1, más alto mejor). MS-SSIM es extensión multi-escala de SSIM. MSE y MAE miden error medio cuadrático y absoluto respectivamente. Índice Perceptual basado en KimiaNet (más bajo mejor).
\\end{table}
"""
        
        # Guardar archivo LaTeX
        latex_file = output_path.replace('.png', '_metrics.tex').replace('.jpg', '_metrics.tex')
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print(f"📄 Tabla LaTeX guardada en: {latex_file}")
        return latex_file
    
    def create_comprehensive_analysis(self, lr_path, predicted_path, hr_path, output_path, image_name="Análisis SR"):
        """
        Crea análisis visual completo de resultados de superresolución
        """
        
        print("🎨 CREANDO ANÁLISIS VISUAL COMPLETO")
        print("=" * 60)
        
        # Cargar imágenes
        print("📁 Cargando imágenes...")
        lr_img = self.load_image(lr_path)
        predicted_img = self.load_image(predicted_path)
        hr_img = self.load_image(hr_path)
        
        if any(img is None for img in [lr_img, predicted_img, hr_img]):
            print("❌ Error cargando alguna de las imágenes")
            return None
        
        print(f"   LR: {lr_img.shape}")
        print(f"   Predicha: {predicted_img.shape}")
        print(f"   HR: {hr_img.shape}")
        
        # Asegurar que HR y predicha tengan el mismo tamaño
        if predicted_img.shape != hr_img.shape:
            print(f"⚠️  Redimensionando HR: {hr_img.shape} → {predicted_img.shape}")
            hr_img = tf.image.resize(hr_img, predicted_img.shape[:2], method='bicubic')
        
        # Calcular factor de escala
        scale_factor = predicted_img.shape[0] // lr_img.shape[0]
        
        # Calcular métricas
        print("📊 Calculando métricas...")
        metrics = self.calculate_metrics(predicted_img, hr_img)
        
        # Mostrar métricas
        print(f"\n📈 MÉTRICAS CALCULADAS:")
        print(f"   PSNR: {metrics['psnr']:.4f} dB")
        print(f"   SSIM: {metrics['ssim']:.4f}")
        print(f"   MS-SSIM: {metrics['ms_ssim']:.4f}")
        print(f"   MSE: {metrics['mse']:.2f}")
        print(f"   MAE: {metrics['mae']:.2f}")
        if metrics['perceptual_index'] is not None:
            print(f"   Índice Perceptual: {metrics['perceptual_index']:.6f}")
        
        # Calcular diferencia absoluta
        abs_diff = self.calculate_absolute_difference(predicted_img, hr_img)
        
        # Generar tabla LaTeX
        self.generate_latex_table(metrics, output_path, image_name)
        
        # Crear visualización SIN tabla embebida
        print("🎨 Generando visualización...")
        
        fig = plt.figure(figsize=(18, 5))  # Más ancho para acomodar el contraste de tamaños
        
        # Convertir tensores a numpy para matplotlib
        def tensor_to_numpy(tensor):
            return np.clip(tensor.numpy().astype(np.uint8), 0, 255)
        
        # Título principal más ajustado
        fig.suptitle(f'Análisis Completo de Superresolución: {image_name}', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        # Configurar subplot con LR MUCHO más pequeña para impacto visual
        gs = GridSpec(1, 5, figure=fig, width_ratios=[0.5, 0.3, 2, 2, 2], 
                     wspace=0.2, left=0.05, right=0.95, top=0.85, bottom=0.05)
        
        # 1. Imagen LR (MUCHO más pequeña visualmente)
        ax1 = fig.add_subplot(gs[0, 0])
        lr_display = tensor_to_numpy(lr_img)
        ax1.imshow(lr_display)
        ax1.set_title(f'LR Input\n{lr_img.shape[0]}×{lr_img.shape[1]}', 
                     fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        # Agregar borde rojo MÁS GRUESO para destacar
        rect = patches.Rectangle((0, 0), lr_img.shape[1]-1, lr_img.shape[0]-1, 
                               linewidth=4, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        
        # Agregar texto "PEQUEÑA" para enfatizar
        ax1.text(0.5, -0.15, '¡Pequeña!', ha='center', va='top', 
                fontsize=9, fontweight='bold', color='red',
                transform=ax1.transAxes)
        
        # 2. Flecha y texto de modelo (espacio intermedio MÁS GRANDE)
        ax_arrow = fig.add_subplot(gs[0, 1])
        ax_arrow.axis('off')
        
        # Flecha más dramática
        arrow = patches.FancyArrowPatch((0.1, 0.5), (0.9, 0.5),
                                       connectionstyle="arc3", 
                                       arrowstyle='->', mutation_scale=30,
                                       transform=ax_arrow.transAxes, 
                                       color='blue', linewidth=4)
        ax_arrow.add_patch(arrow)
        
        # Texto del modelo más llamativo
        ax_arrow.text(0.5, 0.7, f'MODELO\nSR ×{scale_factor}', ha='center', va='center', 
                     fontsize=9, fontweight='bold', color='blue',
                     transform=ax_arrow.transAxes)
        
        # 3. Imagen predicha (GRANDE)
        ax2 = fig.add_subplot(gs[0, 2])
        predicted_display = tensor_to_numpy(predicted_img)
        ax2.imshow(predicted_display)
        ax2.set_title(f'Predicción SR\n{predicted_img.shape[0]}×{predicted_img.shape[1]}', 
                     fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        # Agregar texto "GRANDE" para contrastar
        ax2.text(0.5, -0.08, '¡Grande!', ha='center', va='top', 
                fontsize=10, fontweight='bold', color='green',
                transform=ax2.transAxes)
        
        # 4. Imagen HR original (GRANDE)
        ax3 = fig.add_subplot(gs[0, 3])
        hr_display = tensor_to_numpy(hr_img)
        ax3.imshow(hr_display)
        ax3.set_title(f'HR Original\n{hr_img.shape[0]}×{hr_img.shape[1]}', 
                     fontsize=11, fontweight='bold')
        ax3.axis('off')
        
        # 5. Diferencia absoluta con mapa de calor (GRANDE - mismo tamaño que las otras)
        ax4 = fig.add_subplot(gs[0, 4])
        diff_display = np.mean(abs_diff.numpy(), axis=-1)  # Promedio de canales RGB
        
        im = ax4.imshow(diff_display, cmap='hot', vmin=0, vmax=0.3)
        ax4.set_title(f'Diferencia Absoluta\nError Promedio: {np.mean(diff_display):.4f}', 
                     fontsize=11, fontweight='bold')
        ax4.axis('off')
        
        # Colorbar para diferencia absoluta (normal, no en columna separada)
        cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('Error Absoluto', rotation=270, labelpad=12, fontsize=9)
        
        # NO agregar texto de métricas - esto causaba el espacio en blanco
        
        # Ajustar layout compacto
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Dejar espacio solo para el título
        
        # Guardar imagen
        print(f"💾 Guardando visualización en: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close()
        
        # Resumen en consola
        print("\n" + "=" * 60)
        print("📊 RESUMEN DEL ANÁLISIS")
        print("=" * 60)
        
        # Determinar factor de escala
        scale_factor = predicted_img.shape[0] // lr_img.shape[0]
        print(f"🔍 Factor de escala detectado: ×{scale_factor}")
        print(f"📏 Resolución: {lr_img.shape[0]}×{lr_img.shape[1]} → {predicted_img.shape[0]}×{predicted_img.shape[1]}")
        
        # Evaluación de calidad
        if metrics['psnr'] > 25:
            quality = "Excelente"
        elif metrics['psnr'] > 22:
            quality = "Buena"
        elif metrics['psnr'] > 18:
            quality = "Moderada"
        else:
            quality = "Baja"
        
        print(f"⭐ Calidad estimada: {quality}")
        print(f"💡 Error promedio: {np.mean(diff_display):.4f}")
        
        if metrics['perceptual_index'] is not None:
            if metrics['perceptual_index'] < 0.01:
                perceptual_quality = "Excelente fidelidad biológica"
            elif metrics['perceptual_index'] < 0.05:
                perceptual_quality = "Buena fidelidad biológica"
            else:
                perceptual_quality = "Fidelidad biológica moderada"
            print(f"🧠 Evaluación perceptual: {perceptual_quality}")
        
        print("=" * 60)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Analizador completo de resultados de superresolución")
    
    parser.add_argument("--lr", required=True, help="Imagen LR de entrada")
    parser.add_argument("--predicted", required=True, help="Imagen predicha por el modelo")
    parser.add_argument("--hr", required=True, help="Imagen HR original (ground truth)")
    parser.add_argument("--output", required=True, help="Ruta de salida para el análisis")
    parser.add_argument("--kimianet_weights", 
                       default="/workspace/model-kimianet/KimiaNetKerasWeights.h5",
                       help="Pesos KimiaNet para índice perceptual")
    parser.add_argument("--name", default="Análisis SR", help="Nombre para el análisis")
    
    args = parser.parse_args()
    
    # Verificar que todos los archivos existen
    required_files = [args.lr, args.predicted, args.hr]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ No existe el archivo: {file_path}")
            return 1
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Crear analizador
        analyzer = SuperResolutionAnalyzer(args.kimianet_weights)
        
        # Realizar análisis
        metrics = analyzer.create_comprehensive_analysis(
            lr_path=args.lr,
            predicted_path=args.predicted,
            hr_path=args.hr,
            output_path=args.output,
            image_name=args.name
        )
        
        if metrics:
            print(f"✅ Análisis completado: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"💥 Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())