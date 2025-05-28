#!/usr/bin/env python3
"""
Script simple para usar el modelo ESRGAN exportado (SavedModel)
Este script funciona con el modelo que ya exportaste exitosamente.
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import os

def enhance_image(model_path, input_image_path, output_image_path):
    """
    Mejora una imagen usando el modelo ESRGAN exportado
    
    Args:
        model_path: Ruta al SavedModel exportado
        input_image_path: Ruta de la imagen de entrada (baja resolución)
        output_image_path: Ruta donde guardar la imagen mejorada
    """
    print(f"Cargando modelo ESRGAN desde: {model_path}")
    
    # Cargar el modelo SavedModel
    model = tf.saved_model.load(model_path)
    print("✓ Modelo cargado correctamente")
    
    # Cargar y procesar imagen de entrada
    print(f"Procesando imagen: {input_image_path}")
    img = Image.open(input_image_path).convert('RGB')
    
    # Convertir a tensor
    img_array = np.array(img).astype(np.float32)
    img_tensor = tf.expand_dims(img_array, 0)  # Añadir dimensión de batch
    
    print(f"Forma de entrada: {img_tensor.shape}")
    
    # Aplicar el modelo
    print("Aplicando ESRGAN...")
    enhanced = model(img_tensor)
    
    # Procesar resultado
    enhanced = tf.squeeze(enhanced, 0)  # Remover dimensión de batch
    enhanced = tf.clip_by_value(enhanced, 0, 255)  # Asegurar rango válido
    enhanced = tf.cast(enhanced, tf.uint8)
    
    # Guardar imagen mejorada
    enhanced_img = Image.fromarray(enhanced.numpy())
    enhanced_img.save(output_image_path)
    
    print(f"✓ Imagen mejorada guardada en: {output_image_path}")
    print(f"Resolución original: {img.size}")
    print(f"Resolución mejorada: {enhanced_img.size}")
    
    return enhanced_img

def main():
    parser = argparse.ArgumentParser(description="Mejorar imagen con ESRGAN")
    parser.add_argument("--model", 
                       default="exported_models_128to512/esrgan_microscopy_savedmodel",
                       help="Ruta al modelo SavedModel")
    parser.add_argument("--input", required=True, 
                       help="Imagen de entrada (baja resolución)")
    parser.add_argument("--output", 
                       help="Imagen de salida (alta resolución)")
    
    args = parser.parse_args()
    
    # Generar nombre de salida automáticamente si no se proporciona
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}_enhanced_esrgan.png"
    
    # Verificar que los archivos existen
    if not os.path.exists(args.model):
        print(f"✗ Error: Modelo no encontrado en {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"✗ Error: Imagen no encontrada en {args.input}")
        return
    
    try:
        enhance_image(args.model, args.input, args.output)
        print("✅ ¡Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"✗ Error durante el procesamiento: {e}")

if __name__ == "__main__":
    main()