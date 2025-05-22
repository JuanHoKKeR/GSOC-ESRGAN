import os
import argparse
import tensorflow as tf
from functools import partial
from collections import OrderedDict
import numpy as np

# Definiciones del modelo (mismas que en compare_sr.py)
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
        self._beta = 0.2
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
    """Generador ESRGAN"""
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


def load_model_from_checkpoint(checkpoint_path):
    """Carga el modelo desde checkpoints"""
    print(f"Cargando modelo desde checkpoints: {checkpoint_path}")
    
    # Crear el generador
    generator = RRDBNet(out_channel=3)
    
    # Inicializar con entrada dummy
    dummy_input = tf.random.normal([1, 128, 128, 3])
    generator(dummy_input)
    
    # Restaurar desde checkpoint
    checkpoint = tf.train.Checkpoint(G=generator)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    
    if latest_checkpoint is None:
        raise ValueError(f"No se encontró checkpoint en {checkpoint_path}")
    
    status = checkpoint.restore(latest_checkpoint)
    print(f"✓ Modelo cargado desde: {latest_checkpoint}")
    
    return generator


def export_to_savedmodel(model, output_path):
    """Exporta a formato SavedModel (.pb)"""
    print(f"Exportando a SavedModel: {output_path}")
    
    # Crear un modelo con input/output específicos para inference
    input_layer = tf.keras.Input(shape=[None, None, 3], name="lr_image")
    output = model(input_layer, training=False)
    inference_model = tf.keras.Model(inputs=input_layer, outputs=output, name="esrgan_inference")
    
    # Guardar en formato SavedModel
    tf.saved_model.save(inference_model, output_path)
    print(f"✓ SavedModel guardado en: {output_path}")
    
    # Verificar el modelo guardado
    print("\nInformación del modelo guardado:")
    loaded = tf.saved_model.load(output_path)
    print(f"Signatures: {list(loaded.signatures.keys())}")
    
    return output_path


def export_to_h5(model, output_path):
    """Exporta a formato Keras H5"""
    print(f"Exportando a formato H5: {output_path}")
    
    # Crear modelo funcional para H5
    input_layer = tf.keras.Input(shape=[None, None, 3], name="lr_image")
    output = model(input_layer, training=False)
    h5_model = tf.keras.Model(inputs=input_layer, outputs=output, name="esrgan_x4h5")
    
    # Guardar en formato H5
    h5_model.save(output_path, save_format='h5')
    print(f"✓ Modelo H5 guardado en: {output_path}")
    
    return output_path


def export_to_tflite(model, output_path, quantize=False):
    """Exporta a formato TensorFlow Lite"""
    print(f"Exportando a TensorFlow Lite: {output_path}")
    
    # Crear función concreta para conversión
    @tf.function
    def inference_func(x):
        return model(x, training=False)
    
    # Crear especificación de entrada
    input_spec = tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.float32)
    concrete_func = inference_func.get_concrete_function(input_spec)
    
    # Configurar convertidor
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    if quantize:
        print("Aplicando cuantización...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Dataset representativo para cuantización
        def representative_dataset():
            for _ in range(100):
                data = np.random.random((1, 128, 128, 3)).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    # Convertir
    try:
        tflite_model = converter.convert()
        
        # Guardar
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ Modelo TFLite guardado en: {output_path}")
        print(f"Tamaño del archivo: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"✗ Error al convertir a TFLite: {e}")
        return None


def export_weights_only(model, output_path):
    """Exporta solo los pesos del modelo"""
    print(f"Exportando pesos: {output_path}")
    model.save_weights(output_path)
    print(f"✓ Pesos guardados en: {output_path}")
    return output_path


def test_exported_model(model_path, model_type):
    """Prueba el modelo exportado"""
    print(f"\nProbando modelo exportado: {model_path}")
    
    # Crear imagen de prueba
    test_image = np.random.randint(0, 255, (1, 128, 128, 3)).astype(np.float32)
    
    try:
        if model_type == 'savedmodel':
            model = tf.saved_model.load(model_path)
            result = model(test_image)
            
        elif model_type == 'h5':
            model = tf.keras.models.load_model(model_path)
            result = model(test_image)
            
        elif model_type == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Para TFLite cuantizado, convertir entrada
            if input_details[0]['dtype'] == np.uint8:
                test_image = test_image.astype(np.uint8)
            
            interpreter.set_tensor(input_details[0]['index'], test_image)
            interpreter.invoke()
            result = interpreter.get_tensor(output_details[0]['index'])
            
        print(f"✓ Prueba exitosa! Forma de salida: {result.shape}")
        print(f"Rango de valores: [{np.min(result):.2f}, {np.max(result):.2f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error en la prueba: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Exportar modelo ESRGAN a diferentes formatos")
    parser.add_argument("--checkpoint_path", required=True, help="Ruta a los checkpoints del modelo")
    parser.add_argument("--output_dir", default="exported_models", help="Directorio de salida")
    parser.add_argument("--model_name", default="esrgan_microscopy", help="Nombre base del modelo")
    parser.add_argument("--formats", nargs='+', 
                       choices=['savedmodel', 'h5', 'tflite', 'tflite_quantized', 'weights'],
                       default=['savedmodel', 'h5'],
                       help="Formatos a exportar")
    parser.add_argument("--test", action='store_true', help="Probar modelos exportados")
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar modelo desde checkpoints
    try:
        model = load_model_from_checkpoint(args.checkpoint_path)
        print("✓ Modelo cargado correctamente")
    except Exception as e:
        print(f"✗ Error cargando modelo: {e}")
        return
    
    exported_models = {}
    
    # Exportar en los formatos solicitados
    if 'savedmodel' in args.formats:
        path = os.path.join(args.output_dir, f"{args.model_name}_savedmodel")
        try:
            exported_models['savedmodel'] = export_to_savedmodel(model, path)
        except Exception as e:
            print(f"✗ Error exportando SavedModel: {e}")
    
    if 'h5' in args.formats:
        path = os.path.join(args.output_dir, f"{args.model_name}.h5")
        try:
            exported_models['h5'] = export_to_h5(model, path)
        except Exception as e:
            print(f"✗ Error exportando H5: {e}")
    
    if 'tflite' in args.formats:
        path = os.path.join(args.output_dir, f"{args.model_name}.tflite")
        try:
            exported_models['tflite'] = export_to_tflite(model, path, quantize=False)
        except Exception as e:
            print(f"✗ Error exportando TFLite: {e}")
    
    if 'tflite_quantized' in args.formats:
        path = os.path.join(args.output_dir, f"{args.model_name}_quantized.tflite")
        try:
            exported_models['tflite_quantized'] = export_to_tflite(model, path, quantize=True)
        except Exception as e:
            print(f"✗ Error exportando TFLite cuantizado: {e}")
    
    if 'weights' in args.formats:
        path = os.path.join(args.output_dir, f"{args.model_name}_weights")
        try:
            exported_models['weights'] = export_weights_only(model, path)
        except Exception as e:
            print(f"✗ Error exportando pesos: {e}")
    
    # Mostrar resumen
    print("\n" + "="*50)
    print("RESUMEN DE EXPORTACIÓN")
    print("="*50)
    
    for format_name, path in exported_models.items():
        if path and os.path.exists(path):
            if os.path.isfile(path):
                size = os.path.getsize(path) / (1024*1024)
                print(f"✓ {format_name.upper()}: {path} ({size:.2f} MB)")
            else:
                # Para directorios (SavedModel)
                total_size = sum(os.path.getsize(os.path.join(path, f)) 
                               for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
                size = total_size / (1024*1024)
                print(f"✓ {format_name.upper()}: {path} ({size:.2f} MB)")
    
    # Probar modelos si se solicita
    if args.test:
        print("\n" + "="*50)
        print("PROBANDO MODELOS EXPORTADOS")
        print("="*50)
        
        for format_name, path in exported_models.items():
            if path and os.path.exists(path):
                test_exported_model(path, format_name.split('_')[0])  # Remove '_quantized'


if __name__ == "__main__":
    main()