import os
import argparse
import re
import random

def extract_info(filepath):
    """Extrae información del nombre del archivo para emparejar imágenes usando el número de parche"""
    # Extraer el nombre del archivo de la ruta completa
    filename = os.path.basename(filepath)
    
    # Buscar el patrón que incluye las coordenadas x, y y el número de parche
    # Formato esperado: *_x####_y####_Patch###.jpg
    match = re.search(r'_x(\d+)_y(\d+)_Patch(\d+)\.jpg$', filename)
    if match:
        x_coord = match.group(1)
        y_coord = match.group(2)
        patch_num = match.group(3)
        
        # Crear una clave única basada en coordenadas y número de parche
        unique_key = f"x{x_coord}_y{y_coord}_Patch{patch_num}"
        return unique_key
    return None

def create_meta_pairs(hr_meta_file, lr_meta_file, output_dir, sample_size=None):
    """Crea archivos meta_info con pares LR-HR consistentes basados en el número de parche"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Leer archivos meta_info
    hr_paths = []
    with open(hr_meta_file, 'r') as f:
        hr_paths = [line.strip() for line in f if line.strip()]
    
    lr_paths = []
    with open(lr_meta_file, 'r') as f:
        lr_paths = [line.strip() for line in f if line.strip()]
    
    print(f"HR images: {len(hr_paths)}")
    print(f"LR images: {len(lr_paths)}")
    
    # Organizar imágenes por su identificador único de parche
    hr_dict = {}
    for path in hr_paths:
        unique_key = extract_info(path)
        if unique_key:
            hr_dict[unique_key] = path
    
    lr_dict = {}
    for path in lr_paths:
        unique_key = extract_info(path)
        if unique_key:
            lr_dict[unique_key] = path
    
    # Encontrar identificadores comunes (parches que existen en ambas resoluciones)
    common_keys = set(hr_dict.keys()).intersection(set(lr_dict.keys()))
    print(f"Found {len(common_keys)} common patches")
    
    # Crear pares de imágenes
    pairs = []
    for key in common_keys:
        lr_path = lr_dict[key]
        hr_path = hr_dict[key]
        pairs.append((lr_path, hr_path))
    
    print(f"Created {len(pairs)} exact LR-HR pairs based on patch number")
    
    # Muestra aleatoria si se solicita
    if sample_size and sample_size < len(pairs):
        print(f"Sampling {sample_size} pairs randomly")
        random.shuffle(pairs)
        pairs = pairs[:sample_size]
    
    # Escribir archivos meta_info de pares
    paired_hr_meta = os.path.join(output_dir, "paired_hr_meta.txt")
    paired_lr_meta = os.path.join(output_dir, "paired_lr_meta.txt")
    
    with open(paired_hr_meta, 'w') as f:
        for pair in pairs:
            f.write(f"{pair[1]}\n")
    
    with open(paired_lr_meta, 'w') as f:
        for pair in pairs:
            f.write(f"{pair[0]}\n")
    
    print(f"Saved paired meta_info files:")
    print(f"HR: {paired_hr_meta}")
    print(f"LR: {paired_lr_meta}")
    
    # Mostrar algunos ejemplos de pares para verificación
    if pairs:
        print("\nEjemplos de pares emparejados:")
        for i in range(min(3, len(pairs))):
            print(f"LR: {os.path.basename(pairs[i][0])}")
            print(f"HR: {os.path.basename(pairs[i][1])}")
            print()
    
    return paired_hr_meta, paired_lr_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crear pares consistentes de imágenes LR-HR desde archivos meta_info")
    parser.add_argument("--hr_meta", required=True, help="Archivo meta_info de imágenes de alta resolución")
    parser.add_argument("--lr_meta", required=True, help="Archivo meta_info de imágenes de baja resolución")
    parser.add_argument("--output_dir", default="paired_meta", help="Directorio de salida para los archivos meta_info emparejados")
    parser.add_argument("--sample_size", type=int, default=None, help="Tamaño de la muestra (opcional, para entrenar con un subconjunto)")
    
    args = parser.parse_args()
    create_meta_pairs(args.hr_meta, args.lr_meta, args.output_dir, args.sample_size)