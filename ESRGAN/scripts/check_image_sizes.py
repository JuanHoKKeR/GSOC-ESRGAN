#!/usr/bin/env python3
import os
from PIL import Image
import numpy as np
from collections import Counter

def analyze_image_sizes(meta_file, label=""):
    """Analiza los tamaños de imágenes en un archivo meta_info"""
    print(f"\n=== Análisis de imágenes {label} ===")
    
    sizes = []
    corrupted = 0
    
    with open(meta_file, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    
    print(f"Total de imágenes: {len(paths)}")
    
    # Analizar primeras 100 imágenes para muestra
    sample_size = min(100, len(paths))
    print(f"Analizando muestra de {sample_size} imágenes...")
    
    for i, path in enumerate(paths[:sample_size]):
        try:
            if os.path.exists(path):
                with Image.open(path) as img:
                    sizes.append(img.size)  # (width, height)
            else:
                print(f"Archivo no encontrado: {path}")
                corrupted += 1
        except Exception as e:
            print(f"Error procesando {path}: {e}")
            corrupted += 1
    
    if sizes:
        # Estadísticas
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        
        print(f"Imágenes analizadas: {len(sizes)}")
        print(f"Imágenes corruptas/faltantes: {corrupted}")
        
        print(f"\nAncho:")
        print(f"  Min: {min(widths)}, Max: {max(widths)}")
        print(f"  Promedio: {np.mean(widths):.1f}")
        
        print(f"\nAlto:")
        print(f"  Min: {min(heights)}, Max: {max(heights)}")
        print(f"  Promedio: {np.mean(heights):.1f}")
        
        # Tamaños más comunes
        size_counts = Counter(sizes)
        print(f"\nTamaños más comunes:")
        for size, count in size_counts.most_common(5):
            percentage = (count / len(sizes)) * 100
            print(f"  {size[0]}x{size[1]}: {count} imágenes ({percentage:.1f}%)")
        
        # Verificar si necesita redimensionamiento
        unique_sizes = len(set(sizes))
        print(f"\nTamaños únicos encontrados: {unique_sizes}")
        
        if unique_sizes == 1:
            print("✅ Todas las imágenes tienen el mismo tamaño")
        else:
            print("⚠️  Las imágenes tienen diferentes tamaños - necesita redimensionamiento")
    
    return sizes

def main():
    """Función principal para analizar tu dataset"""
    
    # Rutas a tus archivos meta_info
    hr_meta_file = 'datasets/paired_meta/paired_hr_meta.txt'
    lr_meta_file = 'datasets/paired_meta/paired_lr_meta.txt'
    
    # Verificar que los archivos existen
    if not os.path.exists(hr_meta_file):
        print(f"❌ No se encontró: {hr_meta_file}")
        return
    
    if not os.path.exists(lr_meta_file):
        print(f"❌ No se encontró: {lr_meta_file}")
        return
    
    # Analizar ambos datasets
    hr_sizes = analyze_image_sizes(hr_meta_file, "HR (Alta Resolución)")
    lr_sizes = analyze_image_sizes(lr_meta_file, "LR (Baja Resolución)")
    
    # Análisis comparativo
    print(f"\n=== ANÁLISIS COMPARATIVO ===")
    
    if hr_sizes and lr_sizes:
        # Verificar proporción 4:1
        hr_avg_w = np.mean([s[0] for s in hr_sizes])
        hr_avg_h = np.mean([s[1] for s in hr_sizes])
        lr_avg_w = np.mean([s[0] for s in lr_sizes])
        lr_avg_h = np.mean([s[1] for s in lr_sizes])
        
        scale_w = hr_avg_w / lr_avg_w
        scale_h = hr_avg_h / lr_avg_h
        
        print(f"Proporción promedio:")
        print(f"  Ancho: {scale_w:.2f}:1")
        print(f"  Alto: {scale_h:.2f}:1")
        
        if 3.5 <= scale_w <= 4.5 and 3.5 <= scale_h <= 4.5:
            print("✅ Proporción ~4:1 correcta para ESRGAN")
        else:
            print("⚠️  Proporción no es 4:1 - verificar dataset")
    
    # Recomendaciones
    print(f"\n=== RECOMENDACIONES ===")
    
    if hr_sizes:
        most_common_hr = Counter(hr_sizes).most_common(1)[0]
        hr_size = most_common_hr[0]
        hr_percentage = (most_common_hr[1] / len(hr_sizes)) * 100
        
        print(f"Tamaño HR más común: {hr_size[0]}x{hr_size[1]} ({hr_percentage:.1f}%)")
        
        if hr_size == (256, 256):
            print("✅ HR ya está en 256x256 - NO necesita redimensionamiento")
        else:
            print(f"⚠️  HR no está en 256x256 - considera usar {hr_size[0]}x{hr_size[1]}")
    
    if lr_sizes:
        most_common_lr = Counter(lr_sizes).most_common(1)[0]
        lr_size = most_common_lr[0]
        lr_percentage = (most_common_lr[1] / len(lr_sizes)) * 100
        
        print(f"Tamaño LR más común: {lr_size[0]}x{lr_size[1]} ({lr_percentage:.1f}%)")
        
        if lr_size == (128, 128):
            print("✅ LR ya está en 128x128 - NO necesita redimensionamiento")
        else:
            print(f"⚠️  LR no está en 128x128 - considera usar {lr_size[0]}x{lr_size[1]}")

if __name__ == "__main__":
    main()