#!/usr/bin/env python3
"""
Script SIMPLE para comparar UNA imagen generada vs su ground truth
Calcula PSNR + Índice Perceptual usando DenseNet+KimiaNet
Perfecto para comparar x4 directo vs x2→x2
"""

import argparse
import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121


class KimiaNetPerceptualLoss:
    """Índice perceptual usando DenseNet+KimiaNet para histopatología"""
    
    def __init__(self, kimianet_weights_path):
        print("🧠 Cargando DenseNet121 con pesos KimiaNet...")
        
        # Cargar DenseNet121 sin la capa final
        self.densenet = DenseNet121(
            include_top=False, 
            weights=None,  # Sin pesos de ImageNet
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
        # conv4_block6_concat es una buena capa para características semánticas
        try:
            feature_layer = self.densenet.get_layer('conv4_block6_concat')
        except:
            # Si no existe esa capa, usar una alternativa
            try:
                feature_layer = self.densenet.get_layer('conv4_block24_concat')
            except:
                # Como último recurso, usar la salida completa
                feature_layer = self.densenet.layers[-2]  # Antes del GlobalAveragePooling
        
        self.feature_extractor = tf.keras.Model(
            inputs=self.densenet.input,
            outputs=feature_layer.output
        )
        
        # Congelar el modelo  
        for layer in self.feature_extractor.layers:
            layer.trainable = False
            
        print(f"✅ Extractor de características listo: {feature_layer.name}")
    
    def calculate_perceptual_distance(self, img1, img2):
        """
        Calcula distancia perceptual entre dos imágenes usando KimiaNet
        
        Args:
            img1, img2: Imágenes en formato [H, W, 3] con valores [0, 255]
            
        Returns:
            Distancia perceptual (más bajo = más similar)
        """
        # Asegurar que sean tensores float32
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.cast(img2, tf.float32)
        
        # Agregar dimensión de batch si no existe
        if len(img1.shape) == 3:
            img1 = tf.expand_dims(img1, 0)
        if len(img2.shape) == 3:
            img2 = tf.expand_dims(img2, 0)
        
        # Normalizar para DenseNet (ImageNet normalization, pero KimiaNet podría usar diferente)
        # Para histopatología, usar normalización estándar
        img1_norm = (img1 - 127.5) / 127.5  # [-1, 1]
        img2_norm = (img2 - 127.5) / 127.5  # [-1, 1]
        
        # Extraer características
        features1 = self.feature_extractor(img1_norm)
        features2 = self.feature_extractor(img2_norm)
        
        # Calcular distancia L2 entre características
        perceptual_distance = tf.reduce_mean(tf.square(features1 - features2))
        
        return perceptual_distance


def load_image(image_path):
    """Carga una imagen y la convierte a tensor"""
    try:
        # Leer archivo
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32)
        
        return image
    except Exception as e:
        print(f"❌ Error cargando {image_path}: {e}")
        return None


def calculate_psnr(img1, img2):
    """Calcula PSNR entre dos imágenes"""
    return tf.image.psnr(img1, img2, max_val=255.0)


def resize_to_match(img1, img2):
    """Redimensiona img2 para que coincida con img1 si es necesario"""
    if img1.shape != img2.shape:
        print(f"⚠️  Redimensionando: {img2.shape} → {img1.shape}")
        img2 = tf.image.resize(img2, img1.shape[:2], method='bicubic')
    return img2


def compare_images(generated_path, ground_truth_path, kimianet_weights_path=None):
    """
    Compara una imagen generada con su ground truth
    
    Args:
        generated_path: Ruta a la imagen generada
        ground_truth_path: Ruta a la imagen de referencia (ground truth)
        kimianet_weights_path: Ruta a los pesos KimiaNet (opcional)
    """
    
    print("🔍 COMPARACIÓN DE IMÁGENES")
    print("=" * 50)
    
    # Cargar imágenes
    print("📁 Cargando imágenes...")
    generated_img = load_image(generated_path)
    ground_truth_img = load_image(ground_truth_path)
    
    if generated_img is None or ground_truth_img is None:
        print("❌ Error cargando las imágenes")
        return
    
    print(f"   Generada: {generated_path}")
    print(f"   Dimensiones: {generated_img.shape}")
    print(f"   Ground Truth: {ground_truth_path}")
    print(f"   Dimensiones: {ground_truth_img.shape}")
    
    # Ajustar dimensiones si es necesario
    ground_truth_img = resize_to_match(generated_img, ground_truth_img)
    
    print("\n📊 Calculando métricas...")
    
    # 1. PSNR (rápido)
    psnr_value = calculate_psnr(generated_img, ground_truth_img)
    print(f"✅ PSNR calculado: {float(psnr_value):.4f} dB")
    
    # 2. Índice Perceptual con KimiaNet (más lento)
    print("🧠 Calculando índice perceptual con KimiaNet...")
    perceptual_loss = KimiaNetPerceptualLoss(kimianet_weights_path)
    perceptual_distance = perceptual_loss.calculate_perceptual_distance(
        generated_img, ground_truth_img
    )
    
    print("=" * 50)
    print("🏆 RESULTADOS FINALES")
    print("=" * 50)
    print(f"📈 PSNR: {float(psnr_value):.4f} dB")
    print(f"🧠 Índice Perceptual: {float(perceptual_distance):.6f}")
    print("")
    print("💡 Interpretación:")
    print("   • PSNR: Más alto = mejor (típico: 20-35 dB)")
    print("   • Índice Perceptual: Más bajo = mejor (típico: 0.001-0.1)")
    print("=" * 50)
    
    return {
        'psnr': float(psnr_value),
        'perceptual_index': float(perceptual_distance)
    }


def main():
    parser = argparse.ArgumentParser(description="Comparar UNA imagen generada vs ground truth")
    
    parser.add_argument(
        "--generated",
        required=True,
        help="Ruta a la imagen generada"
    )
    
    parser.add_argument(
        "--ground_truth",
        required=True,
        help="Ruta a la imagen ground truth"
    )
    
    parser.add_argument(
        "--kimianet_weights",
        default="/workspace/model-kimianet/KimiaNetKerasWeights.h5",
        help="Ruta a los pesos KimiaNet (default: /workspace/model-kimianet/KimiaNetKerasWeights.h5)"
    )
    
    args = parser.parse_args()
    
    # Verificar que los archivos existen
    if not os.path.exists(args.generated):
        print(f"❌ No existe la imagen generada: {args.generated}")
        return
    
    if not os.path.exists(args.ground_truth):
        print(f"❌ No existe la imagen ground truth: {args.ground_truth}")
        return
    
    # Comparar imágenes
    try:
        results = compare_images(
            generated_path=args.generated,
            ground_truth_path=args.ground_truth,
            kimianet_weights_path=args.kimianet_weights
        )
        
        print("🎉 Comparación completada")
        
    except Exception as e:
        print(f"💥 Error durante la comparación: {e}")
        raise


if __name__ == "__main__":
    main()