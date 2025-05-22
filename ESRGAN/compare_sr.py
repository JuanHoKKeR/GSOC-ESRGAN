import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import time
from functools import partial
from collections import OrderedDict

# Definir el modelo RRDBNet directamente para evitar dependencias
class RDB(tf.keras.layers.Layer):
    """Residual Dense Block Layer"""
    def __init__(self, out_features=32, bias=True, first_call=True):
        super(RDB, self).__init__()
        _create_conv2d = partial(
            tf.keras.layers.Conv2D,
            out_features,
            kernel_size=[3, 3],
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            strides=[1, 1], padding="same", use_bias=bias)
        self._conv2d_layers = {
            "conv_1": _create_conv2d(),
            "conv_2": _create_conv2d(),
            "conv_3": _create_conv2d(),
            "conv_4": _create_conv2d(),
            "conv_5": _create_conv2d()}
        self._lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self._beta = 0.2  # Valor por defecto
        self._first_call = first_call

    def call(self, input_):
        x1 = self._lrelu(self._conv2d_layers["conv_1"](input_))
        x2 = self._lrelu(self._conv2d_layers["conv_2"](
            tf.concat([input_, x1], -1)))
        x3 = self._lrelu(self._conv2d_layers["conv_3"](
            tf.concat([input_, x1, x2], -1)))
        x4 = self._lrelu(self._conv2d_layers["conv_4"](
            tf.concat([input_, x1, x2, x3], -1)))
        x5 = self._conv2d_layers["conv_5"](tf.concat([input_, x1, x2, x3, x4], -1))
        if self._first_call:
            for _, layer in self._conv2d_layers.items():
                for variable in layer.trainable_variables:
                    original_dtype = variable.dtype
                    value = tf.cast(variable, tf.float32)
                    value = 0.1 * value
                    value = tf.cast(value, original_dtype)
                    variable.assign(value)
            self._first_call = False
        return input_ + self._beta * x5


class RRDB(tf.keras.layers.Layer):
    """Residual in Residual Block Layer"""
    def __init__(self, out_features=32, first_call=True):
        super(RRDB, self).__init__()
        self.RDB1 = RDB(out_features, first_call=first_call)
        self.RDB2 = RDB(out_features, first_call=first_call)
        self.RDB3 = RDB(out_features, first_call=first_call)
        self.beta = 0.2

    def call(self, input_):
        out = self.RDB1(input_)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return input_ + self.beta * out


class RRDBNet(tf.keras.Model):
    """Generador ESRGAN independiente"""
    def __init__(self, out_channel, num_features=32, trunk_size=11, growth_channel=32, use_bias=True, first_call=True):
        super(RRDBNet, self).__init__()
        self.rrdb_block = partial(RRDB, growth_channel, first_call=first_call)
        conv = partial(
            tf.keras.layers.Conv2D,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="same",
            use_bias=use_bias)
        conv_transpose = partial(
            tf.keras.layers.Conv2DTranspose,
            kernel_size=[3, 3],
            strides=[2, 2],
            padding="same",
            use_bias=use_bias)
        self.conv_first = conv(filters=num_features)
        self.rdb_trunk = tf.keras.Sequential(
            [self.rrdb_block() for _ in range(trunk_size)])
        self.conv_trunk = conv(filters=num_features)
        # Upsample
        self.upsample1 = conv_transpose(num_features)
        self.upsample2 = conv_transpose(num_features)
        self.conv_last_1 = conv(num_features)
        self.conv_last_2 = conv(out_channel)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, training=None):
        return self.unsigned_call(inputs)

    def unsigned_call(self, input_):
        feature = self.lrelu(self.conv_first(input_))
        trunk = self.conv_trunk(self.rdb_trunk(feature))
        feature = trunk + feature
        feature = self.lrelu(self.upsample1(feature))
        feature = self.lrelu(self.upsample2(feature))
        feature = self.lrelu(self.conv_last_1(feature))
        out = self.conv_last_2(feature)
        return out


def load_trained_esrgan(checkpoint_path):
    """Carga el modelo ESRGAN entrenado desde los checkpoints"""
    print("Cargando modelo ESRGAN entrenado...")
    
    # Crear el generador con la misma configuración
    generator = RRDBNet(out_channel=3)
    
    # Inicializar el modelo con una entrada dummy
    dummy_input = tf.random.normal([1, 128, 128, 3])
    generator(dummy_input)
    
    # Crear checkpoint y restaurar
    checkpoint = tf.train.Checkpoint(G=generator)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    
    if latest_checkpoint is None:
        raise ValueError(f"No se encontró ningún checkpoint en {checkpoint_path}")
    
    status = checkpoint.restore(latest_checkpoint)
    
    print(f"Modelo cargado desde: {latest_checkpoint}")
    return generator


def bicubic_upscale(image, scale_factor=4):
    """Aplica interpolación bicúbica para super-resolución"""
    if len(image.shape) == 4:  # Batch dimension
        image = tf.squeeze(image, 0)
    
    original_shape = tf.shape(image)
    new_height = original_shape[0] * scale_factor
    new_width = original_shape[1] * scale_factor
    
    # Usar interpolación bicúbica de TensorFlow
    upscaled = tf.image.resize(
        image, 
        [new_height, new_width], 
        method='bicubic'
    )
    
    return tf.expand_dims(upscaled, 0)  # Añadir batch dimension


def load_image(image_path, target_size=None):
    """Carga una imagen y la preprocesa"""
    img = Image.open(image_path).convert('RGB')
    
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    
    # Convertir a tensor
    img_array = np.array(img).astype(np.float32)
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.expand_dims(img_tensor, 0)  # Añadir batch dimension
    
    return img_tensor


def save_image(tensor, path):
    """Guarda un tensor como imagen"""
    if len(tensor.shape) == 4:
        tensor = tf.squeeze(tensor, 0)
    
    # Clip y convertir a uint8
    tensor = tf.clip_by_value(tensor, 0, 255)
    tensor = tf.cast(tensor, tf.uint8)
    
    # Convertir a numpy y guardar
    img_array = tensor.numpy()
    img = Image.fromarray(img_array)
    img.save(path)
    return img


def calculate_metrics(img1, img2):
    """Calcula métricas de calidad entre dos imágenes"""
    # Asegurar que las imágenes tengan la misma forma
    if img1.shape != img2.shape:
        print(f"Warning: Shapes don't match: {img1.shape} vs {img2.shape}")
        return None
    
    # PSNR
    psnr = tf.image.psnr(img1, img2, max_val=255.0)
    
    # SSIM
    ssim = tf.image.ssim(img1, img2, max_val=255.0)
    
    # MSE
    mse = tf.reduce_mean(tf.square(img1 - img2))
    
    return {
        'psnr': float(psnr.numpy()),
        'ssim': float(ssim.numpy()),
        'mse': float(mse.numpy())
    }


def create_comparison_plot(lr_img, hr_img, esrgan_img, bicubic_img, output_path, metrics_esrgan=None, metrics_bicubic=None):
    """Crea una comparación visual de los métodos"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Convertir tensors a numpy para matplotlib
    lr_np = tf.squeeze(lr_img).numpy().astype(np.uint8)
    hr_np = tf.squeeze(hr_img).numpy().astype(np.uint8) if hr_img is not None else None
    esrgan_np = tf.squeeze(esrgan_img).numpy().astype(np.uint8)
    bicubic_np = tf.squeeze(bicubic_img).numpy().astype(np.uint8)
    
    # Imagen LR (entrada)
    axes[0, 0].imshow(lr_np)
    axes[0, 0].set_title('Imagen de Baja Resolución (Entrada)', fontsize=12)
    axes[0, 0].axis('off')
    
    # Imagen HR (referencia) o ESRGAN si no hay HR
    if hr_np is not None:
        axes[0, 1].imshow(hr_np)
        axes[0, 1].set_title('Imagen de Alta Resolución (Referencia)', fontsize=12)
    else:
        axes[0, 1].imshow(esrgan_np)
        axes[0, 1].set_title('ESRGAN (sin referencia)', fontsize=12)
    axes[0, 1].axis('off')
    
    # ESRGAN
    title_esrgan = 'ESRGAN'
    if metrics_esrgan:
        title_esrgan += f'\nPSNR: {metrics_esrgan["psnr"]:.2f} dB'
        title_esrgan += f'\nSSIM: {metrics_esrgan["ssim"]:.4f}'
    axes[1, 0].imshow(esrgan_np)
    axes[1, 0].set_title(title_esrgan, fontsize=12)
    axes[1, 0].axis('off')
    
    # Bicúbica
    title_bicubic = 'Interpolación Bicúbica'
    if metrics_bicubic:
        title_bicubic += f'\nPSNR: {metrics_bicubic["psnr"]:.2f} dB'
        title_bicubic += f'\nSSIM: {metrics_bicubic["ssim"]:.4f}'
    axes[1, 1].imshow(bicubic_np)
    axes[1, 1].set_title(title_bicubic, fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparación guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comparar ESRGAN entrenado con interpolación bicúbica")
    parser.add_argument("--input_lr", required=True, help="Imagen de baja resolución de entrada")
    parser.add_argument("--input_hr", default=None, help="Imagen de alta resolución de referencia (opcional)")
    parser.add_argument("--checkpoint_path", required=True, help="Ruta a los checkpoints del modelo ESRGAN")
    parser.add_argument("--output_dir", default="comparison_results", help="Directorio de salida")
    parser.add_argument("--scale_factor", type=int, default=4, help="Factor de escala (default: 4)")
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar modelo ESRGAN
    try:
        esrgan_model = load_trained_esrgan(args.checkpoint_path)
        print("✓ Modelo ESRGAN cargado correctamente")
    except Exception as e:
        print(f"✗ Error al cargar modelo ESRGAN: {e}")
        return
    
    # Cargar imagen de baja resolución
    print(f"Cargando imagen LR: {args.input_lr}")
    lr_image = load_image(args.input_lr)
    print(f"Forma de imagen LR: {lr_image.shape}")
    
    # Cargar imagen de alta resolución si está disponible
    hr_image = None
    if args.input_hr and os.path.exists(args.input_hr):
        print(f"Cargando imagen HR de referencia: {args.input_hr}")
        hr_image = load_image(args.input_hr)
        print(f"Forma de imagen HR: {hr_image.shape}")
    
    # ESRGAN Super-resolución
    print("Aplicando ESRGAN...")
    start_time = time.time()
    esrgan_result = esrgan_model(lr_image, training=False)
    esrgan_time = time.time() - start_time
    print(f"✓ ESRGAN completado en {esrgan_time:.2f} segundos")
    print(f"Forma de salida ESRGAN: {esrgan_result.shape}")
    
    # Interpolación bicúbica
    print("Aplicando interpolación bicúbica...")
    start_time = time.time()
    bicubic_result = bicubic_upscale(lr_image, scale_factor=args.scale_factor)
    bicubic_time = time.time() - start_time
    print(f"✓ Interpolación bicúbica completada en {bicubic_time:.2f} segundos")
    print(f"Forma de salida bicúbica: {bicubic_result.shape}")
    
    # Asegurar que ambas imágenes tengan el mismo tamaño
    target_height = min(esrgan_result.shape[1], bicubic_result.shape[1])
    target_width = min(esrgan_result.shape[2], bicubic_result.shape[2])
    
    esrgan_result = tf.image.resize(esrgan_result, [target_height, target_width])
    bicubic_result = tf.image.resize(bicubic_result, [target_height, target_width])
    
    # Calcular métricas si hay imagen de referencia
    metrics_esrgan = None
    metrics_bicubic = None
    
    if hr_image is not None:
        # Redimensionar HR para que coincida
        hr_resized = tf.image.resize(hr_image, [target_height, target_width])
        
        print("Calculando métricas...")
        metrics_esrgan = calculate_metrics(hr_resized, esrgan_result)
        metrics_bicubic = calculate_metrics(hr_resized, bicubic_result)
        
        if metrics_esrgan and metrics_bicubic:
            print("\n=== MÉTRICAS DE COMPARACIÓN ===")
            print(f"ESRGAN - PSNR: {metrics_esrgan['psnr']:.2f} dB, SSIM: {metrics_esrgan['ssim']:.4f}, MSE: {metrics_esrgan['mse']:.2f}")
            print(f"Bicúbica - PSNR: {metrics_bicubic['psnr']:.2f} dB, SSIM: {metrics_bicubic['ssim']:.4f}, MSE: {metrics_bicubic['mse']:.2f}")
            
            # Determinar ganador
            psnr_winner = "ESRGAN" if metrics_esrgan['psnr'] > metrics_bicubic['psnr'] else "Bicúbica"
            ssim_winner = "ESRGAN" if metrics_esrgan['ssim'] > metrics_bicubic['ssim'] else "Bicúbica"
            
            print(f"\nGanador PSNR: {psnr_winner} ({abs(metrics_esrgan['psnr'] - metrics_bicubic['psnr']):.2f} dB de diferencia)")
            print(f"Ganador SSIM: {ssim_winner} ({abs(metrics_esrgan['ssim'] - metrics_bicubic['ssim']):.4f} de diferencia)")
    
    # Guardar imágenes individuales
    base_name = os.path.splitext(os.path.basename(args.input_lr))[0]
    
    save_image(lr_image, os.path.join(args.output_dir, f"{base_name}_lr.png"))
    save_image(esrgan_result, os.path.join(args.output_dir, f"{base_name}_esrgan.png"))
    save_image(bicubic_result, os.path.join(args.output_dir, f"{base_name}_bicubic.png"))
    
    if hr_image is not None:
        save_image(hr_image, os.path.join(args.output_dir, f"{base_name}_hr_reference.png"))
    
    # Crear comparación visual
    comparison_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")
    create_comparison_plot(
        lr_image, hr_image, esrgan_result, bicubic_result, 
        comparison_path, metrics_esrgan, metrics_bicubic
    )
    
    # Guardar métricas en archivo de texto
    if metrics_esrgan and metrics_bicubic:
        metrics_path = os.path.join(args.output_dir, f"{base_name}_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write("=== COMPARACIÓN ESRGAN vs INTERPOLACIÓN BICÚBICA ===\n\n")
            f.write(f"Imagen procesada: {args.input_lr}\n")
            f.write(f"Imagen de referencia: {args.input_hr}\n\n")
            f.write(f"Tiempo ESRGAN: {esrgan_time:.2f} segundos\n")
            f.write(f"Tiempo Bicúbica: {bicubic_time:.2f} segundos\n\n")
            f.write(f"ESRGAN - PSNR: {metrics_esrgan['psnr']:.2f} dB, SSIM: {metrics_esrgan['ssim']:.4f}, MSE: {metrics_esrgan['mse']:.2f}\n")
            f.write(f"Bicúbica - PSNR: {metrics_bicubic['psnr']:.2f} dB, SSIM: {metrics_bicubic['ssim']:.4f}, MSE: {metrics_bicubic['mse']:.2f}\n\n")
            f.write(f"Mejora PSNR: {metrics_esrgan['psnr'] - metrics_bicubic['psnr']:.2f} dB\n")
            f.write(f"Mejora SSIM: {metrics_esrgan['ssim'] - metrics_bicubic['ssim']:.4f}\n")
    
    print(f"\n✓ Comparación completada. Resultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()