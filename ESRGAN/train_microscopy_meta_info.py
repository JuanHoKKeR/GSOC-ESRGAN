import os
import argparse
import re
import random

def extract_info(filename):
    """Extrae información del nombre del archivo para emparejar imágenes"""
    # Extrae el identificador principal de la imagen, utilizando un patrón que capture
    # el identificador común entre imágenes de diferentes resoluciones
    # En este caso, busca el patrón "TCGA-XX-XXXX" que parece ser común en tus imágenes
    match = re.search(r'(TCGA-[A-Z]{2}-\d{4}.*?_\d+x_)(\d+px)', filename)
    if match:
        base_id = match.group(1)  # Identificador base (sin la resolución)
        resolution = match.group(2)  # Resolución (512px, 1024px, etc.)
        return base_id, resolution
    return None, None

def create_meta_pairs(hr_meta_file, lr_meta_file, output_dir, sample_size=None):
    """Crea archivos meta_info con pares LR-HR consistentes"""
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
    
    # Organizar imágenes por su identificador base
    hr_dict = {}
    for path in hr_paths:
        filename = os.path.basename(path)
        base_id, resolution = extract_info(filename)
        if base_id and resolution:
            key = base_id
            if key not in hr_dict:
                hr_dict[key] = []
            hr_dict[key].append(path)
    
    lr_dict = {}
    for path in lr_paths:
        filename = os.path.basename(path)
        base_id, resolution = extract_info(filename)
        if base_id and resolution:
            key = base_id
            if key not in lr_dict:
                lr_dict[key] = []
            lr_dict[key].append(path)
    
    # Encontrar identificadores comunes
    common_ids = set(hr_dict.keys()).intersection(set(lr_dict.keys()))
    print(f"Found {len(common_ids)} common image IDs")
    
    # Crear pares
    pairs = []
    for key in common_ids:
        hr_images = hr_dict[key]
        lr_images = lr_dict[key]
        
        # Emparejar imágenes del mismo ID
        # Si hay más imágenes HR que LR, o viceversa, algunas quedarán sin emparejar
        min_len = min(len(hr_images), len(lr_images))
        
        # Mezclar para selección aleatoria de pares
        random.shuffle(hr_images)
        random.shuffle(lr_images)
        
        for i in range(min_len):
            pairs.append((lr_images[i], hr_images[i]))
    
    print(f"Created {len(pairs)} LR-HR pairs")
    
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
    
    return paired_hr_meta, paired_lr_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crear pares consistentes de imágenes LR-HR desde archivos meta_info")
    parser.add_argument("--hr_meta", required=True, help="Archivo meta_info de imágenes de alta resolución")
    parser.add_argument("--lr_meta", required=True, help="Archivo meta_info de imágenes de baja resolución")
    parser.add_argument("--output_dir", default="paired_meta", help="Directorio de salida para los archivos meta_info emparejados")
    parser.add_argument("--sample_size", type=int, default=None, help="Tamaño de la muestra (opcional, para entrenar con un subconjunto)")
    
    args = parser.parse_args()
    create_meta_pairs(args.hr_meta, args.lr_meta, args.output_dir, args.sample_size)