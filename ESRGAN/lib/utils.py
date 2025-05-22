import os
from functools import partial
import tensorflow as tf
from absl import logging
from lib import settings
from tensorflow.keras.optimizers import Adam

""" Utility functions needed for training ESRGAN model. """

# Checkpoint Utilities


def save_checkpoint(checkpoint, training_phase, basepath=""):
  """ Saves checkpoint.
      Args:
        checkpoint: tf.train.Checkpoint object
        training_phase: The training phase of the model to load/store the checkpoint for.
                        can be one of the two "phase_1" or "phase_2"
        basepath: Base path to load checkpoints from.
  """
  dir_ = settings.Settings()["checkpoint_path"][training_phase]
  if basepath:
    dir_ = os.path.join(basepath, dir_)
  dir_ = os.path.join(dir_, os.path.basename(dir_))
  checkpoint.save(file_prefix=dir_)
  logging.debug("Prefix: %s. checkpoint saved successfully!" % dir_)


def load_checkpoint(checkpoint, training_phase, basepath=""):
  """ Loads checkpoint.
      Args:
        checkpoint: tf.train.Checkpoint object
        training_phase: The training phase of the model to load/store the checkpoint for.
                        can be one of the two "phase_1" or "phase_2"
        basepath: Base Path to load checkpoints from.
      Returns:
        Status object from checkpoint.restore
  """
  logging.info("Loading check point for: %s" % training_phase)
  dir_ = settings.Settings()["checkpoint_path"][training_phase]
  if basepath:
    dir_ = os.path.join(basepath, dir_)
  if tf.io.gfile.exists(os.path.join(dir_, "checkpoint")):
    logging.info("Found checkpoint at: %s" % dir_)
    status = checkpoint.restore(tf.train.latest_checkpoint(dir_))
    return status
  return None  # Retornar explícitamente None cuando no se encuentra checkpoint

# Network Interpolation utility


def interpolate_generator(
    generator_fn,
    discriminator,
    alpha,
    dimension,
    factor=4,
    basepath=""):
  """ Interpolates between the weights of the PSNR model and GAN model

       Refer to Section 3.4 of https://arxiv.org/pdf/1809.00219.pdf (Xintao et. al.)

       Args:
         generator_fn: function which returns the keras model the generator used.
         discriminiator: Keras model of the discriminator.
         alpha: interpolation parameter between both the weights of both the models.
         dimension: dimension of the high resolution image
         factor: scale factor of the model
         basepath: Base directory to load checkpoints from.
       Returns:
         Keras model of a generator with weights interpolated between the PSNR and GAN model.
  """
  assert 0 <= alpha <= 1
  size = dimension
  if not tf.nest.is_nested(dimension):
    size = [dimension, dimension]
  logging.debug("Interpolating generator. Alpha: %f" % alpha)
  optimizer = partial(Adam)
  gan_generator = generator_fn()
  psnr_generator = generator_fn()
  # building generators
  gan_generator(tf.random.normal(
      [1, size[0] // factor, size[1] // factor, 3]))
  psnr_generator(tf.random.normal(
      [1, size[0] // factor, size[1] // factor, 3]))

  phase_1_ckpt = tf.train.Checkpoint(
      G=psnr_generator, G_optimizer=optimizer())
  phase_2_ckpt = tf.train.Checkpoint(
      G=gan_generator,
      G_optimizer=optimizer(),
      D=discriminator,
      D_optimizer=optimizer())

  load_checkpoint(phase_1_ckpt, "phase_1", basepath)
  load_checkpoint(phase_2_ckpt, "phase_2", basepath)

  # Consuming Checkpoint
  logging.debug("Consuming Variables: Adversarial generator") 
  # Verificar que unsigned_call exista, de lo contrario usar call regular
  if hasattr(gan_generator, "unsigned_call"):
    gan_generator.unsigned_call(tf.random.normal(
        [1, size[0] // factor, size[1] // factor, 3]))
  else:
    gan_generator(tf.random.normal(
        [1, size[0] // factor, size[1] // factor, 3]))
  
  input_layer = tf.keras.Input(shape=[None, None, 3], name="input_0")
  output = gan_generator(input_layer)
  gan_generator = tf.keras.Model(
      inputs=[input_layer],
      outputs=[output])

  logging.debug("Consuming Variables: PSNR generator")
  # Verificar que unsigned_call exista, de lo contrario usar call regular
  if hasattr(psnr_generator, "unsigned_call"):
    psnr_generator.unsigned_call(tf.random.normal(
        [1, size[0] // factor, size[1] // factor, 3]))
  else:
    psnr_generator(tf.random.normal(
        [1, size[0] // factor, size[1] // factor, 3]))

  for variables_1, variables_2 in zip(
          gan_generator.trainable_variables, psnr_generator.trainable_variables):
    variables_1.assign((1 - alpha) * variables_2 + alpha * variables_1)

  return gan_generator

# Losses


def preprocess_input(image):
  image = image[..., ::-1]
  mean = -tf.constant([103.939, 116.779, 123.68])
  return tf.nn.bias_add(image, mean)


def PerceptualLoss(weights="imagenet", input_shape=(None, None, 3), loss_type="L1"):
  """ Perceptual Loss using VGG19
      Args:
        weights: Weights to be loaded (default: 'imagenet').
        input_shape: Shape of input image (default: (None, None, 3)).
        loss_type: Loss type for features. (L1 / L2)
      Returns:
        Loss function for perceptual loss
  """
  vgg_model = tf.keras.applications.vgg19.VGG19(
      input_shape=input_shape, weights=weights, include_top=False)
  for layer in vgg_model.layers:
    layer.trainable = False
  
  # Obtener la capa y eliminar la activación
  block5_conv4 = vgg_model.get_layer("block5_conv4")
  # Crear un nuevo modelo con la misma configuración pero sin activación
  config = block5_conv4.get_config()
  config["activation"] = None
  new_block5_conv4 = tf.keras.layers.Conv2D.from_config(config)
  # Copiar los pesos
  new_block5_conv4.build(block5_conv4.input_shape)
  new_block5_conv4.set_weights(block5_conv4.get_weights())
  
  # Crear un nuevo modelo VGG hasta block5_conv4 sin activación
  x = vgg_model.input
  for layer in vgg_model.layers[1:]:  # Saltamos el input_layer
    if layer.name == "block5_conv4":
      x = new_block5_conv4(x)
      break
    else:
      x = layer(x)
  
  phi = tf.keras.Model(inputs=[vgg_model.input], outputs=[x])

  def loss(y_true, y_pred):
    y_true = preprocess_input(y_true)
    y_pred = preprocess_input(y_pred)
    
    if loss_type.lower() == "l1":
      return tf.reduce_mean(tf.abs(phi(y_true) - phi(y_pred)))

    if loss_type.lower() == "l2":
      return tf.reduce_mean(
          tf.reduce_mean(
              (phi(y_true) - phi(y_pred))**2,
              axis=0))
    raise ValueError(
        "Loss Function: \"%s\" not defined for Perceptual Loss" %
        loss_type)
  return loss


def pixel_loss(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  return tf.reduce_mean(tf.reduce_mean(tf.abs(y_true - y_pred), axis=0))


def RelativisticAverageLoss(non_transformed_disc, type_="G"):
  """ Relativistic Average Loss based on RaGAN
      Args:
      non_transformed_disc: non activated discriminator Model
      type_: type of loss to Ra loss to produce.
             'G': Relativistic average loss for generator
             'D': Relativistic average loss for discriminator
      Returns:
        Loss function for the specified type
  """
  loss = None

  def D_Ra(x, y):
    return non_transformed_disc(
        x) - tf.reduce_mean(non_transformed_disc(y))

  def loss_D(y_true, y_pred):
    """
      Relativistic Average Loss for Discriminator
      Args:
        y_true: Real Image
        y_pred: Generated Image
    """
    real_logits = D_Ra(y_true, y_pred)
    fake_logits = D_Ra(y_pred, y_true)
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(real_logits), logits=real_logits))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(fake_logits), logits=fake_logits))
    return real_loss + fake_loss

  def loss_G(y_true, y_pred):
    """
     Relativistic Average Loss for Generator
     Args:
       y_true: Real Image
       y_pred: Generated Image
    """
    real_logits = D_Ra(y_true, y_pred)
    fake_logits = D_Ra(y_pred, y_true)
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(real_logits), logits=real_logits))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(fake_logits), logits=fake_logits))
    return real_loss + fake_loss
  
  if type_ == "G":
    loss = loss_G
  elif type_ == "D":
    loss = loss_D
  else:
    raise ValueError(f"Loss type '{type_}' not supported. Use 'G' or 'D'.")
  
  return loss


# Strategy Utils

def assign_to_worker(use_tpu):
  if use_tpu:
    return "/job:worker"
  return ""


class SingleDeviceStrategy(object):
  """ Strategy wrapper para usar cuando no se usa TPU/multi-GPU """

  def __enter__(self, *args, **kwargs):
    pass

  def __exit__(self, *args, **kwargs):
    pass

  def experimental_distribute_dataset(self, dataset, *args, **kwargs):
    # En TF 2.11, usar distribute_dataset en lugar de experimental_distribute_dataset
    return dataset

  def distribute_datasets_from_function(self, dataset_fn, *args, **kwargs):
    # Añadido para compatibilidad con TF 2.11
    return dataset_fn(*args, **kwargs)
  
  def experimental_run_v2(self, fn, args=(), kwargs=None):
    # Obsoleto - mantener por retrocompatibilidad pero usar run
    if kwargs is None:
      kwargs = {}
    return self.run(fn, args, kwargs)
  
  def run(self, fn, args=(), kwargs=None):
    """Implementación para TF 2.11"""
    if kwargs is None:
        kwargs = {}
    return fn(*args, **kwargs)

  def reduce(self, reduction_type, distributed_data, axis=None):
    return distributed_data

  def scope(self):
    return self


# Model Utils


class RDB(tf.keras.layers.Layer):
  """ Residual Dense Block Layer """

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
    self._beta = settings.Settings()["RDB"].get("residual_scale_beta", 0.2)
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
      logging.debug("Initializing with MSRA")
      for _, layer in self._conv2d_layers.items():
          for variable in layer.trainable_variables:
              # Convertir explícitamente a float32 para la operación y luego de vuelta al tipo original
              original_dtype = variable.dtype
              value = tf.cast(variable, tf.float32)
              value = 0.1 * value
              value = tf.cast(value, original_dtype)
              variable.assign(value)
      self._first_call = False
    return input_ + self._beta * x5


class RRDB(tf.keras.layers.Layer):
  """ Residual in Residual Block Layer """

  def __init__(self, out_features=32, first_call=True):
    super(RRDB, self).__init__()
    self.RDB1 = RDB(out_features, first_call=first_call)
    self.RDB2 = RDB(out_features, first_call=first_call)
    self.RDB3 = RDB(out_features, first_call=first_call)
    self.beta = settings.Settings()["RDB"].get("residual_scale_beta", 0.2)

  def call(self, input_):
    out = self.RDB1(input_)
    out = self.RDB2(out)
    out = self.RDB3(out)
    return input_ + self.beta * out


def calculate_ssim(y_true, y_pred):
    """Calcula SSIM entre la imagen real y la generada"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255.0))

def calculate_ms_ssim(y_true, y_pred):
    """Calcula MS-SSIM entre la imagen real y la generada"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=255.0))

def calculate_psnr(y_true, y_pred):
    """Calcula PSNR entre la imagen real y la generada"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=255.0))

def calculate_mse(y_true, y_pred):
    """Calcula MSE entre la imagen real y la generada"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.square(y_true - y_pred))

def load_pretrained_generator(generator, pretrained_path):
    """Carga pesos preentrenados en el generador RRDBNet
    
    Args:
        generator: Instancia de RRDBNet para cargar los pesos
        pretrained_path: Ruta al directorio del modelo SavedModel de ESRGAN
        
    Returns:
        El generador con los pesos preentrenados cargados
    """
    logging.info(f"Cargando pesos preentrenados desde: {pretrained_path}")
    
    try:
        # Asegurarse de que el generador está inicializado
        if not generator.built:
            dummy_input = tf.random.normal([1, 128, 128, 3])
            generator(dummy_input)
            logging.info("Inicializado generador con pesos aleatorios")
        
        # Corregir la ruta si se está apuntando directamente al archivo saved_model.pb
        model_dir = pretrained_path
        if pretrained_path.endswith('saved_model.pb'):
            model_dir = os.path.dirname(pretrained_path)
            logging.info(f"Ajustando ruta de modelo a directorio: {model_dir}")
        
        # Verificar que el directorio existe y contiene un SavedModel
        if not os.path.exists(os.path.join(model_dir, 'saved_model.pb')) and not os.path.exists(os.path.join(model_dir, 'saved_model.pbtxt')):
            logging.warning(f"No se encontró saved_model.pb ni saved_model.pbtxt en {model_dir}")
            # Buscar recursivamente en subdirectorios
            for root, dirs, files in os.walk(model_dir):
                if 'saved_model.pb' in files:
                    model_dir = root
                    logging.info(f"Encontrado saved_model.pb en: {model_dir}")
                    break
        
        # Cargar el modelo preentrenado
        pretrained_model = tf.saved_model.load(model_dir)
        logging.info("Modelo preentrenado cargado correctamente")
        
        # Imprimir estructura de variables para depuración
        logging.debug("Variables en el generador:")
        gen_var_names = [v.name for v in generator.variables]
        for name in gen_var_names[:5]:  # Mostrar las primeras 5 variables
            logging.debug(f"  {name}")
        
        logging.debug("Variables en el modelo preentrenado:")
        pretrained_var_names = [v.name for v in pretrained_model.variables]
        for name in pretrained_var_names[:5]:  # Mostrar las primeras 5 variables
            logging.debug(f"  {name}")
        
        # Crear mapa de variables
        var_mapping = {}
        for gen_var in generator.variables:
            # Extraer el nombre sin prefijo del modelo y sin sufijo de índice
            var_name = gen_var.name.split('/')[-1].split(':')[0]
            var_mapping[var_name] = gen_var
        
        # Transferir pesos
        transferred_count = 0
        for pretrained_var in pretrained_model.variables:
            # Obtener nombre simplificado de la variable preentrenada
            pre_var_name = pretrained_var.name.split('/')[-1].split(':')[0]
            
            # Buscar la variable correspondiente en el generador
            if pre_var_name in var_mapping:
                gen_var = var_mapping[pre_var_name]
                
                # Verificar compatibilidad de formas
                if gen_var.shape == pretrained_var.shape:
                    gen_var.assign(pretrained_var)
                    transferred_count += 1
                else:
                    logging.warning(f"Forma incompatible para {pre_var_name}: {gen_var.shape} vs {pretrained_var.shape}")
        
        logging.info(f"Se transfirieron {transferred_count} variables de {len(generator.variables)}")
        
        # Validar que la carga funcionó
        test_input = tf.random.normal([1, 128, 128, 3])
        test_output = generator(test_input)
        logging.info(f"Prueba de inferencia exitosa. Forma de salida: {test_output.shape}")
        
        return generator
    
    except Exception as e:
        logging.error(f"Error al cargar pesos preentrenados: {e}", exc_info=True)
        logging.info("Continuando con inicialización aleatoria")
        return generator