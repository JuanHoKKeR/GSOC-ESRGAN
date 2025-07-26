#!/usr/bin/env python3
"""
Script para crear visualización comparativa de métodos de superresolución
Compara: x16 directo, cascada x2x2x2x2, interpolación bicúbica vs ground truth
Genera tabla LaTeX por separado para insertar en el paper
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from PIL import Image, ImageFont, ImageDraw
import cv2

# Configurar fuente Computer Modern Roman para matplotlib
plt.rcParams.update({
    'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False
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


def load_image(image_path, target_size=None):
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


def calculate_metrics(img1, img2):
    """Calcula PSNR entre dos imágenes"""
    psnr = tf.image.psnr(img1, img2, max_val=255.0)
    ssim = tf.image.ssim(img1, img2, max_val=255.0)
    return float(psnr), float(ssim)


def create_bicubic_upscale(lr_image_path, scale_factor=16):
    """Crea una versión interpolada bicúbica"""
    print(f"🔄 Creando interpolación bicúbica x{scale_factor}...")
    
    # Cargar imagen LR
    lr_img = load_image(lr_image_path)
    if lr_img is None:
        return None
    
    # Obtener dimensiones originales
    h, w = lr_img.shape[:2]
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # Aplicar interpolación bicúbica
    bicubic_img = tf.image.resize(lr_img, [new_h, new_w], method='bicubic')
    
    return bicubic_img


def generate_latex_table(metrics_results, output_path):
    """
    Genera tabla LaTeX con las métricas de comparación
    """
    latex_table = """\\begin{table*}[!htb]
\\centering
\\caption{Comparación de métodos de superresolución para imágenes histopatológicas: escalado 64×64 → 1024×1024}
\\label{tab:sr_methods_comparison}
\\makebox[\\textwidth]{%
\\scriptsize
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Método de} & \\textbf{PSNR} & \\textbf{SSIM} & \\textbf{Índice} & \\textbf{Calidad} \\\\
\\textbf{Superresolución} & \\textbf{(dB)} & & \\textbf{Perceptual} & \\textbf{Estimada} \\\\
\\hline
"""
    
    # Agregar filas de datos
    for method_name, metrics in metrics_results.items():
        # Mapear nombres para LaTeX
        method_latex = method_name.replace("x2⁴", "x2$^4$").replace("x16", "x16")
        
        # Determinar calidad estimada
        if metrics['psnr'] > 24 and metrics['perceptual'] < 0.02:
            quality = "Excelente"
        elif metrics['psnr'] > 22:
            quality = "Buena"
        else:
            quality = "Moderada"
        
        latex_table += f"{method_latex} & {metrics['psnr']:.2f} & {metrics['ssim']:.3f} & {metrics['perceptual']:.4f} & {quality} \\\\\n"
    
    latex_table += """\\hline
\\end{tabular}
}
\\textbf{Notas:} PSNR más alto indica mejor fidelidad pixel-wise. SSIM evalúa similitud estructural (0-1, más alto mejor). Índice Perceptual basado en KimiaNet específico para histopatología (más bajo mejor). La cascada x2$^4$ aplica cuatro modelos ×2 secuencialmente, mientras que x16 directo usa un solo modelo de factor 16.
\\end{table*}
\\clearpage % Fuerza que la tabla se coloque antes de continuar
"""
    
    # Guardar archivo LaTeX
    latex_file = output_path.replace('.png', '_table.tex').replace('.jpg', '_table.tex')
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"📄 Tabla LaTeX guardada en: {latex_file}")
    return latex_file


def create_comparison_visualization(
    lr_path,
    cascade_path,
    direct_x16_path,
    ground_truth_path,
    kimianet_weights_path,
    output_path
):
    """
    Crea visualización comparativa de todos los métodos de superresolución
    """
    
    print("🎨 CREANDO VISUALIZACIÓN COMPARATIVA")
    print("=" * 60)
    
    # Inicializar evaluador perceptual
    perceptual_evaluator = KimiaNetPerceptualLoss(kimianet_weights_path)
    
    # Cargar todas las imágenes
    print("📁 Cargando imágenes...")
    lr_img = load_image(lr_path)
    cascade_img = load_image(cascade_path)
    direct_img = load_image(direct_x16_path)
    gt_img = load_image(ground_truth_path, target_size=(1024, 1024))
    bicubic_img = create_bicubic_upscale(lr_path, scale_factor=16)
    
    if any(img is None for img in [lr_img, cascade_img, direct_img, gt_img, bicubic_img]):
        print("❌ Error cargando alguna de las imágenes")
        return
    
    # Redimensionar LR para visualización
    lr_display = tf.image.resize(lr_img, [256, 256], method='nearest')  # Nearest para mantener pixelado
    
    print("📊 Calculando métricas...")
    
    # Calcular métricas para cada método
    methods = {
        "Cascada x2⁴": cascade_img,
        "Directo x16": direct_img,
        "Bicúbica x16": bicubic_img
    }
    
    metrics_results = {}
    
    for method_name, img in methods.items():
        print(f"   Evaluando: {method_name}")
        
        # PSNR y SSIM
        psnr, ssim = calculate_metrics(img, gt_img)
        
        # Índice perceptual
        perceptual_dist = perceptual_evaluator.calculate_perceptual_distance(img, gt_img)
        
        metrics_results[method_name] = {
            'psnr': psnr,
            'ssim': ssim,
            'perceptual': float(perceptual_dist)
        }
        
        print(f"      PSNR: {psnr:.4f} dB")
        print(f"      SSIM: {ssim:.4f}")
        print(f"      Perceptual: {float(perceptual_dist):.6f}")
    
    # Generar tabla LaTeX
    generate_latex_table(metrics_results, output_path)
    
    # Crear la visualización
    print("🎨 Generando visualización...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 1])
    
    # Convertir tensores a numpy para matplotlib
    def tensor_to_numpy(tensor):
        return np.clip(tensor.numpy().astype(np.uint8), 0, 255)
    
    fig.suptitle('Comparación de Métodos de Superresolución: 64×64 → 1024×1024', 
                 fontsize=20, fontweight='bold', y=0.96)
    
    # Fila superior: Proceso de escalado
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(tensor_to_numpy(lr_display))
    ax1.set_title('Imagen Original\n64×64', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Flecha hacia cascada
    ax_arrow1 = fig.add_subplot(gs[0, 1])
    ax_arrow1.text(0.5, 0.7, '→ x2 → x2 → x2 → x2', ha='center', va='center', 
                   fontsize=10, fontweight='bold', transform=ax_arrow1.transAxes)
    ax_arrow1.text(0.5, 0.3, '(Cascada)', ha='center', va='center', 
                   fontsize=9, style='italic', transform=ax_arrow1.transAxes)
    ax_arrow1.axis('off')
    
    # Flecha hacia directo
    ax_arrow2 = fig.add_subplot(gs[0, 2])
    ax_arrow2.text(0.5, 0.7, '→ x16', ha='center', va='center', 
                   fontsize=10, fontweight='bold', transform=ax_arrow2.transAxes)
    ax_arrow2.text(0.5, 0.3, '(Directo)', ha='center', va='center', 
                   fontsize=9, style='italic', transform=ax_arrow2.transAxes)
    ax_arrow2.axis('off')
    
    # Flecha hacia bicúbica
    ax_arrow3 = fig.add_subplot(gs[0, 3])
    ax_arrow3.text(0.5, 0.7, '→ Bicúbica x16', ha='center', va='center', 
                   fontsize=10, fontweight='bold', transform=ax_arrow3.transAxes)
    ax_arrow3.text(0.5, 0.3, '(Interpolación)', ha='center', va='center', 
                   fontsize=9, style='italic', transform=ax_arrow3.transAxes)
    ax_arrow3.axis('off')
    
    # Ground Truth
    ax_gt_ref = fig.add_subplot(gs[0, 4])
    ax_gt_ref.imshow(tensor_to_numpy(gt_img))
    ax_gt_ref.set_title('Ground Truth\n1024×1024', fontsize=12, fontweight='bold')
    ax_gt_ref.axis('off')
    
    # Fila inferior: Resultados
    images = [cascade_img, direct_img, bicubic_img, gt_img]
    titles = ['Cascada x2⁴', 'Directo x16', 'Bicúbica x16', 'Ground Truth']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[1, i+1])
        ax.imshow(tensor_to_numpy(img))
        
        # Agregar métricas en el título (excepto para ground truth)
        if title != 'Ground Truth':
            metrics = metrics_results[title]
            title_with_metrics = f"{title}\nPSNR: {metrics['psnr']:.2f} dB\nSSIM: {metrics['ssim']:.3f}\nPerceptual: {metrics['perceptual']:.4f}"
        else:
            title_with_metrics = title
            
        ax.set_title(title_with_metrics, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Guardar imagen
    print(f"💾 Guardando visualización en: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Mostrar resumen en consola
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 60)
    
    # Encontrar el mejor método para cada métrica
    best_psnr = max(metrics_results.items(), key=lambda x: x[1]['psnr'])
    best_ssim = max(metrics_results.items(), key=lambda x: x[1]['ssim'])
    best_perceptual = min(metrics_results.items(), key=lambda x: x[1]['perceptual'])
    
    print(f"🏆 Mejor PSNR: {best_psnr[0]} ({best_psnr[1]['psnr']:.2f} dB)")
    print(f"🏆 Mejor SSIM: {best_ssim[0]} ({best_ssim[1]['ssim']:.3f})")
    print(f"🏆 Mejor Perceptual: {best_perceptual[0]} ({best_perceptual[1]['perceptual']:.4f})")
    
    print("\n💡 Conclusiones:")
    if best_psnr[0] == best_perceptual[0]:
        print(f"   • {best_psnr[0]} es superior en calidad objetiva y perceptual")
    else:
        print(f"   • {best_psnr[0]} tiene mejor calidad objetiva (PSNR)")
        print(f"   • {best_perceptual[0]} tiene mejor calidad perceptual")
    
    print("=" * 60)
    
    return metrics_results


def main():
    parser = argparse.ArgumentParser(description="Crear visualización comparativa de métodos de superresolución")
    
    parser.add_argument("--lr", required=True, help="Imagen LR original (64x64)")
    parser.add_argument("--cascade", required=True, help="Resultado de cascada x2x2x2x2")
    parser.add_argument("--direct", required=True, help="Resultado directo x16")
    parser.add_argument("--ground_truth", required=True, help="Ground truth (1024x1024)")
    parser.add_argument("--kimianet_weights", 
                       default="/workspace/model-kimianet/KimiaNetKerasWeights.h5",
                       help="Pesos KimiaNet")
    parser.add_argument("--output", required=True, help="Ruta de salida para la visualización")
    
    args = parser.parse_args()
    
    # Verificar que todos los archivos existen
    required_files = [args.lr, args.cascade, args.direct, args.ground_truth]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ No existe el archivo: {file_path}")
            return
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Crear visualización
        results = create_comparison_visualization(
            lr_path=args.lr,
            cascade_path=args.cascade,
            direct_x16_path=args.direct,
            ground_truth_path=args.ground_truth,
            kimianet_weights_path=args.kimianet_weights,
            output_path=args.output
        )
        
        print(f"✅ Visualización completada: {args.output}")
        
    except Exception as e:
        print(f"💥 Error durante la visualización: {e}")
        raise


if __name__ == "__main__":
    main()