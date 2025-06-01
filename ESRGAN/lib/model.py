from collections import OrderedDict
from functools import partial
import tensorflow as tf
from lib import utils
from tensorflow.keras.applications import DenseNet121
import os


""" Keras Models for ESRGAN
    Classes:
      RRDBNet: Generator of ESRGAN. (Residual in Residual Network)
      VGGArch: VGG28 Architecture making the Discriminator ESRGAN
      DenseNetDiscriminator: Alternative discriminator based on DenseNet121
"""


class RRDBNet(tf.keras.Model):
  """ Residual in Residual Network consisting of:
      - Convolution Layers
      - Residual in Residual Block as the trunk of the model
      - Pixel Shuffler layers (tf.nn.depth_to_space)
      - Upscaling Convolutional Layers

      Args:
        out_channel: number of channels of the fake output image.
        num_features (default: 32): number of filters to use in the convolutional layers.
        trunk_size (default: 3): number of Residual in Residual Blocks to form the trunk.
        growth_channel (default: 32): number of filters to use in the internal convolutional layers.
        use_bias (default: True): boolean to indicate if bias is to be used in the conv layers.
        first_call (default: True): boolean to initialize weights on first call.
  """

  def __init__(
          self,
          out_channel,
          num_features=32,
          trunk_size=11,
          growth_channel=32,
          use_bias=True,
          first_call=True):
    super(RRDBNet, self).__init__()
    self.rrdb_block = partial(utils.RRDB, growth_channel, first_call=first_call)
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
    #self.upsample2 = conv_transpose(num_features) # Scale x4
    #self.upsample3 = conv_transpose(num_features) # Scale x8
    self.conv_last_1 = conv(num_features)
    self.conv_last_2 = conv(out_channel)
    self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
    
    # Para compatibilidad con TF 2.11, se mantendrá el tracking de ambos métodos
    self._supports_tf_logs = True

  # El decorador @tf.function puede mejorar el rendimiento durante entrenamiento e inferencia
  # Descomenta esta línea para mejorar el rendimiento
  # @tf.function
  def call(self, inputs, training=None):
    """Método estándar de llamada de Keras
    
    Args:
        inputs: Tensor de entrada con imagen de baja resolución
        training: Booleano que indica si está en modo entrenamiento
        
    Returns:
        Tensor con la imagen generada de alta resolución
    """
    return self.unsigned_call(inputs)

  def unsigned_call(self, input_):
    """Implementación interna de la inferencia
    
    Args:
        input_: Tensor de entrada con imagen de baja resolución
        
    Returns:
        Tensor con la imagen generada de alta resolución
    """
    feature = self.lrelu(self.conv_first(input_))
    trunk = self.conv_trunk(self.rdb_trunk(feature))
    feature = trunk + feature
    feature = self.lrelu(
            self.upsample1(feature))
    #feature = self.lrelu(self.upsample2(feature)) # Scale x4
    #feature = self.lrelu(self.upsample3(feature)) # Scale x8
    feature = self.lrelu(self.conv_last_1(feature))
    out = self.conv_last_2(feature)
    return out


class VGGArch(tf.keras.Model):
  """ Keras Model for VGG28 Architecture needed to form
      the discriminator of the architecture.
      
      Args:
        output_shape (default: 1): output_shape of the generator
        num_features (default: 64): number of features to be used in the convolutional layers
                                    a factor of 2**i will be multiplied as per the need
        use_bias (default: True): Boolean to indicate whether to use biases for convolution layers
  """

  def __init__(
          self,
          batch_size=8,
          output_shape=1,
          num_features=64,
          use_bias=False):

    super(VGGArch, self).__init__()
    conv = partial(
        tf.keras.layers.Conv2D,
        kernel_size=[3, 3], use_bias=use_bias, padding="same")
    batch_norm = partial(tf.keras.layers.BatchNormalization)
    
    # Esta función no se usa y se mantiene por compatibilidad
    # def no_batch_norm(x): return x
    
    self._lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
    self._dense_1 = tf.keras.layers.Dense(100)
    self._dense_2 = tf.keras.layers.Dense(output_shape)
    self._conv_layers = OrderedDict()
    self._batch_norm = OrderedDict()
    self._conv_layers["conv_0_0"] = conv(filters=num_features, strides=1)
    self._conv_layers["conv_0_1"] = conv(filters=num_features, strides=2)
    self._batch_norm["bn_0_1"] = batch_norm()
    for i in range(1, 4):
      for j in range(1, 3):
        self._conv_layers["conv_%d_%d" % (i, j)] = conv(
            filters=num_features * (2**i), strides=j)
        self._batch_norm["bn_%d_%d" % (i, j)] = batch_norm()
    
    # Para compatibilidad con TF 2.11, se mantendrá el tracking de ambos métodos
    self._supports_tf_logs = True

  # @tf.function  # Descomenta esta línea para mejorar el rendimiento
  def call(self, inputs, training=None):
    """Método estándar de llamada de Keras
    
    Args:
        inputs: Tensor de entrada con imagen 
        training: Booleano que indica si está en modo entrenamiento
        
    Returns:
        Tensor con la clasificación (real/falso)
    """
    return self.unsigned_call(inputs)
    
  def unsigned_call(self, input_):
    """Implementación interna de la inferencia
    
    Args:
        input_: Tensor de entrada con imagen
        
    Returns:
        Tensor con la clasificación (real/falso)
    """
    features = self._lrelu(self._conv_layers["conv_0_0"](input_))
    features = self._lrelu(
        self._batch_norm["bn_0_1"](
            self._conv_layers["conv_0_1"](features)))
    # VGG Trunk
    for i in range(1, 4):
      for j in range(1, 3):
        features = self._lrelu(
            self._batch_norm["bn_%d_%d" % (i, j)](
                self._conv_layers["conv_%d_%d" % (i, j)](features)))

    flattened = tf.keras.layers.Flatten()(features)
    dense = self._lrelu(self._dense_1(flattened))
    out = self._dense_2(dense)
    return out

class DenseNetDiscriminator(tf.keras.Model):
    """Discriminador basado en DenseNet121 con pesos de KimiaNet
    
    Args:
        output_shape (default: 1): Forma de salida, generalmente 1 para discriminador
        use_bias (default: True): Si usar bias en las capas densas
        kimianet_weights_path (default: None): Ruta al archivo de pesos KimiaNet
    """
    
    def __init__(self, output_shape=1, use_bias=True, kimianet_weights_path=None):
        super(DenseNetDiscriminator, self).__init__()
        
        # Cargar DenseNet121 sin la capa final de clasificación
        self.base_model = DenseNet121(include_top=False, weights=None, input_shape=(None, None, 3))
        
        # Cargar pesos de KimiaNet si se proporciona la ruta
        if kimianet_weights_path:
            if os.path.exists(kimianet_weights_path):
                self.base_model.load_weights(kimianet_weights_path)
            else:
                print(f"Advertencia: No se encontró el archivo de pesos en {kimianet_weights_path}")
        
        # Congelar algunas capas base
        for layer in self.base_model.layers[:100]:
          layer.trainable = False
        
        # Global Average Pooling para manejar tamaños variables
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        
        # Capas densas para la clasificación final
        self.dense1 = tf.keras.layers.Dense(256)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(output_shape)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        
        # Para compatibilidad con TF 2.11, se mantendrá el tracking de ambos métodos
        self._supports_tf_logs = True
        
    # @tf.function  # Descomenta esta línea para mejorar el rendimiento
    def call(self, inputs, training=None):
        """Método estándar de llamada de Keras
        
        Args:
            inputs: Tensor de entrada con imagen
            training: Booleano que indica si está en modo entrenamiento
            
        Returns:
            Tensor con la clasificación (real/falso)
        """
        return self.unsigned_call(inputs, training=training)
        
    def unsigned_call(self, input_, training=None):
        """Implementación interna de la inferencia
        
        Args:
            input_: Tensor de entrada con imagen
            
        Returns:
            Tensor con la clasificación (real/falso)
        """
        # Procesar mediante DenseNet121/KimiaNet
        features = self.base_model(input_, training=training)
        
        # Global pooling para manejar diferentes resoluciones
        pooled = self.global_pool(features)
        
        # Capas finales de clasificación
        x = self.leaky_relu(self.dense1(pooled))
        x = self.dropout(x, training=training)
        output = self.dense2(x)
        
        return output
    
class OptimizedVGGArch(tf.keras.Model):
  """ Versión optimizada del discriminador VGG para usar menos memoria """

  def __init__(self, batch_size=8, output_shape=1, num_features=32, use_bias=False):  # Reducido num_features
    super(OptimizedVGGArch, self).__init__()
    conv = partial(
        tf.keras.layers.Conv2D,
        kernel_size=[3, 3], use_bias=use_bias, padding="same")
    batch_norm = partial(tf.keras.layers.BatchNormalization)
    
    self._lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
    self._dense_1 = tf.keras.layers.Dense(50)  # Reducido de 100 a 50
    self._dense_2 = tf.keras.layers.Dense(output_shape)
    self._conv_layers = OrderedDict()
    self._batch_norm = OrderedDict()
    
    # Añadir downsampling inicial para reducir resolución rápidamente
    self._initial_downsample = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
    
    self._conv_layers["conv_0_0"] = conv(filters=num_features, strides=1)
    self._conv_layers["conv_0_1"] = conv(filters=num_features, strides=2)
    self._batch_norm["bn_0_1"] = batch_norm()
    
    # Reducir número de capas
    for i in range(1, 3):  # Reducido de 4 a 3
      for j in range(1, 3):
        self._conv_layers["conv_%d_%d" % (i, j)] = conv(
            filters=num_features * (2**i), strides=j)
        self._batch_norm["bn_%d_%d" % (i, j)] = batch_norm()

  def call(self, inputs, training=None):
    return self.unsigned_call(inputs)
    
  def unsigned_call(self, input_):
    # Downsample inicialmente para reducir memoria
    input_ = self._initial_downsample(input_)
    
    features = self._lrelu(self._conv_layers["conv_0_0"](input_))
    features = self._lrelu(
        self._batch_norm["bn_0_1"](
            self._conv_layers["conv_0_1"](features)))
    
    # VGG Trunk reducido
    for i in range(1, 3):  # Reducido
      for j in range(1, 3):
        features = self._lrelu(
            self._batch_norm["bn_%d_%d" % (i, j)](
                self._conv_layers["conv_%d_%d" % (i, j)](features)))

    flattened = tf.keras.layers.Flatten()(features)
    dense = self._lrelu(self._dense_1(flattened))
    out = self._dense_2(dense)
    return out