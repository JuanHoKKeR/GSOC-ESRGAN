import os
import sys
import tensorflow as tf
import numpy as np

# Verificar versiones
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

# Verificar GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs disponibles: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")

# Habilitar crecimiento de memoria
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth habilitado para {gpu}")
    except Exception as e:
        print(f"Error al configurar memory growth: {e}")

# Probar operación básica en GPU
if len(gpus) > 0:
    print("\nEjecutando test básico en GPU...")
    with tf.device('/GPU:0'):
        # Crear tensores grandes para probar rendimiento
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        
        # Medir tiempo
        import time
        start = time.time()
        c = tf.matmul(a, b)
        # Forzar ejecución
        _ = c.numpy().mean()
        end = time.time()
        
        print(f"Operación completada en {end - start:.2f} segundos")
        print(f"Forma del tensor resultante: {c.shape}")
else:
    print("\n¡ADVERTENCIA! No se detectaron GPUs. TensorFlow está usando CPU.")

# Verificar disponibilidad de mixed precision
print("\nVerificando soporte para mixed precision...")
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    policy = mixed_precision.global_policy()
    print(f"Mixed precision policy configurada: {policy}")
except Exception as e:
    print(f"Error al configurar mixed precision: {e}")

print("\nVerificación completada.")