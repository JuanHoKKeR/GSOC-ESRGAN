import os
import random
import numpy as np
from absl import logging
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds

""" Dataset Handlers for ESRGAN """


def scale_down(method="bicubic", dimension=256, size=None, factor=4):
  """ Scales down function based on the parameters provided.
      Args:
        method (default: bicubic): Interpolation method to be used for Scaling down the image.
        dimension (default: 256): Dimension of the high resolution counterpart.
        size (default: None): [height, width] of the image.
        factor (default: 4): Factor by which the model enhances the low resolution image.
      Returns:
        tf.data.Dataset mappable python function based on the configuration.
  """
  if not size:
    size = (dimension, dimension)
  size_ = {"size": size}

  def scale_fn(image, *args, **kwargs):
    size = size_["size"]
    high_resolution = image
    if not kwargs.get("no_random_crop", None):
      high_resolution = tf.image.random_crop(
          image, [size[0], size[1], image.shape[-1]])

    low_resolution = tf.image.resize(
        high_resolution,
        [size[0] // factor, size[1] // factor],
        method=method)
    low_resolution = tf.clip_by_value(low_resolution, 0, 255)
    high_resolution = tf.clip_by_value(high_resolution, 0, 255)
    return low_resolution, high_resolution
  scale_fn.size = size_["size"]
  return scale_fn


def augment_image(
        brightness_delta=0.05,
        contrast_factor=[0.7, 1.3],
        saturation=[0.6, 1.6]):
  """ Helper function used for augmentation of images in the dataset.
      Args:
        brightness_delta: maximum value for randomly assigning brightness of the image.
        contrast_factor: list / tuple of minimum and maximum value of factor to set random contrast.
                          None, if not to be used.
        saturation: list / tuple of minimum and maximum value of factor to set random saturation.
                    None, if not to be used.
      Returns:
        tf.data.Dataset mappable function for image augmentation
  """
  def augment_fn(low_resolution, high_resolution, *args, **kwargs):
    # Augmenting data (~ 80%)
    def augment_steps_fn(low_resolution, high_resolution):
      # Randomly rotating image (~50%)
      def rotate_fn(low_resolution, high_resolution):
        times = tf.random.uniform(minval=1, maxval=4, dtype=tf.int32, shape=[])
        return (tf.image.rot90(low_resolution, times),
                tf.image.rot90(high_resolution, times))
      low_resolution, high_resolution = tf.cond(
          tf.less_equal(tf.random.uniform([]), 0.5),
          lambda: rotate_fn(low_resolution, high_resolution),
          lambda: (low_resolution, high_resolution))
      # Randomly flipping image (~50%)
      def flip_fn(low_resolution, high_resolution):
        return (tf.image.flip_left_right(low_resolution),
                tf.image.flip_left_right(high_resolution))
      low_resolution, high_resolution = tf.cond(
          tf.less_equal(tf.random.uniform([]), 0.5),
          lambda: flip_fn(low_resolution, high_resolution),
          lambda: (low_resolution, high_resolution))

      # Randomly setting brightness of image (~50%)
      def brightness_fn(low_resolution, high_resolution):
        delta = tf.random.uniform(minval=0, maxval=brightness_delta, dtype=tf.float32, shape=[])
        return (tf.image.adjust_brightness(low_resolution, delta=delta),
                tf.image.adjust_brightness(high_resolution, delta=delta))
      low_resolution, high_resolution = tf.cond(
          tf.less_equal(tf.random.uniform([]), 0.5),
          lambda: brightness_fn(low_resolution, high_resolution),
          lambda: (low_resolution, high_resolution))

      # Randomly setting constrast (~50%)
      def contrast_fn(low_resolution, high_resolution):
        factor = tf.random.uniform(
            minval=contrast_factor[0],
            maxval=contrast_factor[1],
            dtype=tf.float32, shape=[])
        return (tf.image.adjust_contrast(low_resolution, factor),
                tf.image.adjust_contrast(high_resolution, factor))
      if contrast_factor:
        low_resolution, high_resolution = tf.cond(
            tf.less_equal(tf.random.uniform([]), 0.5),
            lambda: contrast_fn(low_resolution, high_resolution),
            lambda: (low_resolution, high_resolution))

      # Randomly setting saturation(~50%)
      def saturation_fn(low_resolution, high_resolution):
        factor = tf.random.uniform(
            minval=saturation[0],
            maxval=saturation[1],
            dtype=tf.float32,
            shape=[])
        return (tf.image.adjust_saturation(low_resolution, factor),
               tf.image.adjust_saturation(high_resolution, factor))
      if saturation:
        low_resolution, high_resolution = tf.cond(
            tf.less_equal(tf.random.uniform([]), 0.5),
            lambda: saturation_fn(low_resolution, high_resolution),
            lambda: (low_resolution, high_resolution))

      return low_resolution, high_resolution

    # Randomly returning unchanged data (~20%)
    return tf.cond(
        tf.less_equal(tf.random.uniform([]), 0.2),
        lambda: (low_resolution, high_resolution),
        partial(augment_steps_fn, low_resolution, high_resolution))

  return augment_fn


def reform_dataset(dataset, types, size, num_elems=None):
  """ Helper function to convert the output_dtype of the dataset
      from (tf.float32, tf.uint8) to desired dtype
      Args:
        dataset: Source dataset(image-label dataset) to convert.
        types: tuple / list of target datatype.
        size: [height, width] threshold of the images.
        num_elems: Number of Data points to store
      Returns:
        tf.data.Dataset with the images of dimension >= Args.size and types = Args.types
  """
  _carrier = {"num_elems": num_elems}

  def generator_fn():
    for idx, data in enumerate(dataset, 1):
      if _carrier["num_elems"]:
        if idx >= _carrier["num_elems"]:
          break  # Usar break en lugar de StopIteration para mejor compatibilidad
      if data[0].shape[0] >= size[0] and data[0].shape[1] >= size[1]:
        yield data[0], data[1]
      else:
        continue
  return tf.data.Dataset.from_generator(
      generator_fn,
      output_signature=(
          tf.TensorSpec(shape=[None, None, 3], dtype=types[0]),
          tf.TensorSpec(shape=None, dtype=types[1])
      )
)

def load_div2k_dataset(
    hr_directory,
    lr_directory,
    hr_size,
    batch_size=None,
    repeat=0,
    shuffle=False,
    augment=False,
    cache="cache/",
    buffer_size=3*32,
    options=None):
  """ Loads Div2K dataset """
  scale = os.path.basename(lr_directory).lower()
  lr_size=[hr_size[0] // int(scale[1:]), hr_size[1] // int(scale[1:])]
  def _load_fn(hr_files):
    for image in hr_files:
      hr_image = tf.io.decode_image(tf.io.read_file(image))
      lr_name = os.path.basename(image).split(".")
      lr_name[-2] = lr_name[-2] + scale
      lr_name = ".".join(lr_name)
      lr_image = tf.io.decode_image(
          tf.io.read_file(
              os.path.join(lr_directory, lr_name)))
      for lr_h in range(0, lr_image.shape[0] - lr_size[0] + 1, 40):
        for lr_w in range(0, lr_image.shape[1] - lr_size[1] + 1, 40):
          hr_h = lr_h * int(scale[1:])
          hr_w = lr_w * int(scale[1:])
          low_res = tf.image.crop_to_bounding_box(lr_image, lr_h, lr_w, lr_size[0], lr_size[1])
          high_res = tf.image.crop_to_bounding_box(hr_image, hr_h, hr_w, hr_size[0], hr_size[1])
          yield tf.cast(low_res, tf.float32), tf.cast(high_res, tf.float32)
  
  hr_files = tf.io.gfile.glob(os.path.join(hr_directory, "*.jpg"))
  hr_files.extend(tf.io.gfile.glob(os.path.join(hr_directory, "*.png")))
  dataset = tf.data.Dataset.from_generator(
      partial(_load_fn, hr_files),
      (tf.float32, tf.float32),
      (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])))
  if shuffle:
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
  if repeat:
    dataset = dataset.repeat(repeat)
  if augment:
    dataset = dataset.map(
        augment_image(saturation=None),
        num_parallel_calls=tf.data.AUTOTUNE)  # Cambiado de experimental.AUTOTUNE a AUTOTUNE
  if batch_size:
    dataset = dataset.batch(batch_size, drop_remainder=True)
  if options:
    dataset = dataset.with_options(options)
  return dataset

def load_dataset_directory(
        name,
        directory,
        low_res_map_fn,
        batch_size=None,
        shuffle=False,
        augment=False,
        cache_dir="cache/",
        buffer_size=3 * 32,
        options=None,
        num_elems=65536):
  """ Loads image_label dataset from a local directory:
      Structure of the local directory should be:

      dataset_name
      |__ label1
      |   |__ image1
      |   |__ image2
      |
      |__ label2
          |__ image1
          |__ image2

      Args:
          name: Name of the dataset.
          directory: Location where the manual directory is located
          low_res_map_fn: tf.data.Dataset mappable function to generate
                          (low_resolution, high_resolution) pair
          batch_size: Size of batch to create
          shuffle: Boolean to indicate if data is to be shuffled.
          augment: Boolean to indicate if data is to augmented.
          cache_dir: Cache directory to save the data to.
          buffer_size: size of shuffle buffer to use.
          num_elems: Number of elements to iterate over in the dataset.
      Returns:
          A tf.data.Dataset having data as (low_resolution, high_resolution)
  """
  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.mkdir(cache_dir)
  dl_config = tfds.download.DownloadConfig(manual_dir=directory)
  dataset = reform_dataset(
      tfds.load(
          "image_label_folder/dataset_name=%s" %
          name,
          split="train",
          as_supervised=True,
          download_and_prepare_kwargs={
              "download_config": dl_config}),
      (tf.float32, tf.float32),
      size=low_res_map_fn.size,
      num_elems=num_elems)
  if options:
    dataset = dataset.with_options(options)
  dataset = dataset.map(
      low_res_map_fn,
      num_parallel_calls=tf.data.AUTOTUNE)  # Cambiado de experimental.AUTOTUNE a AUTOTUNE
  if batch_size:
    dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size)
  # .cache(cache_dir))

  if shuffle:
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

  if augment:
    dataset = dataset.map(
        augment_image(saturation=None),
        num_parallel_calls=tf.data.AUTOTUNE)  # Cambiado de experimental.AUTOTUNE a AUTOTUNE
  return dataset


def load_dataset(
        name,
        low_res_map_fn,
        split="train",
        batch_size=None,
        shuffle=True,
        augment=True,
        buffer_size=3 * 32,
        cache_dir="cache/",
        data_dir=None,
        options=None,
        num_elems=65536):
  """ Helper function to load a dataset from tensorflow_datasets
      Args:
          name: Name of the dataset builder from tensorflow_datasets to load the data.
          low_res_map_fn: tf.data.Dataset mappable function to generate
                          (low_resolution, high_resolution) pair.
          split: split of the dataset to return.
          batch_size: Size of batch to create
          shuffle: Boolean to indicate if data is to be shuffled.
          augment: Boolean to indicate if data is to augmented.
          buffer_size: size of shuffle buffer to use.
          cache_dir: Cache directory to save the data to.
          data_dir: Directory to save the downloaded dataset to.
          num_elems: Number of elements to iterate over in the dataset.
      Returns:
          A tf.data.Dataset having data as (low_resolution, high_resolution)

  """
  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.mkdir(cache_dir)
  dataset = reform_dataset(
      tfds.load(
          name,
          data_dir=data_dir,
          split=split,
          as_supervised=True),
      (tf.float32, tf.float32),
      size=low_res_map_fn.size,
      num_elems=num_elems)
  if options:
    dataset = dataset.with_options(options)
  dataset = dataset.map(
      low_res_map_fn,
      num_parallel_calls=tf.data.AUTOTUNE)  # Cambiado de experimental.AUTOTUNE a AUTOTUNE
  if batch_size:
    dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size)
  # .cache(cache_dir))

  if shuffle:
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

  if augment:
    dataset = dataset.map(
        augment_image(saturation=None),
        num_parallel_calls=tf.data.AUTOTUNE)  # Cambiado de experimental.AUTOTUNE a AUTOTUNE
  return dataset


def load_tfrecord_dataset(
        tfrecord_path, lr_size, hr_size,
        shuffle=True, shuffle_buffer=128, batch_size=None):  # Añadido batch_size como parámetro opcional
  """ Loads TFRecords for feeding to ESRGAN
      Args:
        tfrecord_path: Path to load .tfrecord files from.
        lr_size: size of the low_resolution images.
        hr_size: size of the high resolution images.
        shuffle: Boolean to indicate if the data will be shuffled(default: True)
        shuffle_buffer: Size of the shuffle buffer to use (default: 128)
        batch_size: Size of batch to create (default: None)
  """
  def _parse_tf_record(serialized_example):
    """ Parses Single Serialized Tensor from TFRecord """
    features = {
        "low_res_image": tf.io.FixedLenFeature([], dtype=tf.string),
        "high_res_image": tf.io.FixedLenFeature([], dtype=tf.string)}
    example = tf.io.parse_single_example(serialized_example, features)
    lr_image = tf.io.parse_tensor(
        example["low_res_image"],
        out_type=tf.float32)
    lr_image = tf.reshape(lr_image, lr_size)
    hr_image = tf.io.parse_tensor(
        example["high_res_image"],
        out_type=tf.float32)
    hr_image = tf.reshape(hr_image, hr_size)
    return lr_image, hr_image
  files = tf.io.gfile.glob(
      os.path.join(tfrecord_path, "*.tfrecord"))
  if len(files) == 0:
    raise ValueError("Path Doesn't contain any file")
  ds = tf.data.TFRecordDataset(files).map(
      _parse_tf_record, num_parallel_calls=tf.data.AUTOTUNE)  # Cambiado a AUTOTUNE
  
  if len(files) == 1:
    option = tf.data.Options()
    option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF  # Actualizado a API de TF 2.11
    ds = ds.with_options(option)
  
  if shuffle:
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
  
  # Añadir batch si se proporciona
  if batch_size:
    ds = ds.batch(batch_size, drop_remainder=True)
  
  # Prefetch para mejor rendimiento
  ds = ds.prefetch(tf.data.AUTOTUNE)
  
  return ds


def load_dataset_from_meta_info(
        hr_meta_file,
        lr_meta_file,
        base_path="",
        transforms=None,
        batch_size=16,
        buffer_size=1,
        repeat=True,
        shuffle=True,
        lr_size=(256, 256),
        hr_size=(512, 512),
        num_parallel_calls=16):
    """Carga datos desde archivos meta_info que contienen rutas de imágenes.
    
    Args:
        hr_meta_file: Ruta al archivo meta_info con imágenes HR
        lr_meta_file: Ruta al archivo meta_info con imágenes LR
        base_path: Ruta base para añadir a las rutas en los archivos meta_info
        transforms: Función opcional para transformar las imágenes
        batch_size: Tamaño del batch
        buffer_size: Tamaño del buffer para shuffle
        repeat: Si se debe repetir el dataset indefinidamente
        shuffle: Si se debe barajar el dataset
        lr_size: Tamaño objetivo para imágenes de baja resolución (altura, ancho)
        hr_size: Tamaño objetivo para imágenes de alta resolución (altura, ancho)
        
    Returns:
        Un tf.data.Dataset con pares (LR, HR) de imágenes
    """
    # No importar dentro de la función - ya está en el ámbito principal
    # import tensorflow as tf
    # import os
    # from absl import logging
    
    # Cargar rutas de archivos de meta_info
    hr_paths = []
    with open(hr_meta_file, 'r') as f:
        hr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
    
    lr_paths = []
    with open(lr_meta_file, 'r') as f:
        lr_paths = [os.path.join(base_path, line.strip()) for line in f if line.strip()]
    
    logging.info(f"Cargadas {len(hr_paths)} imágenes HR y {len(lr_paths)} imágenes LR")
    
    # Verificar las primeras rutas para depuración
    if hr_paths:
        logging.debug(f"Primera ruta HR: {hr_paths[0]}")
        logging.debug(f"Existe archivo: {os.path.exists(hr_paths[0])}")
    if lr_paths:
        logging.debug(f"Primera ruta LR: {lr_paths[0]}")
        logging.debug(f"Existe archivo: {os.path.exists(lr_paths[0])}")
    
    # Verificar que tenemos suficientes imágenes
    if len(hr_paths) == 0 or len(lr_paths) == 0:
        raise ValueError(f"No se encontraron imágenes en los archivos meta_info: {hr_meta_file}, {lr_meta_file}")
    
    # Verificar que el número de imágenes coincide
    if len(hr_paths) != len(lr_paths):
        logging.warning(f"Número diferente de imágenes HR ({len(hr_paths)}) y LR ({len(lr_paths)})")
    
    # Crear TF Dataset
    lr_ds = tf.data.Dataset.from_tensor_slices(lr_paths)
    hr_ds = tf.data.Dataset.from_tensor_slices(hr_paths)
    
    # Función para cargar y procesar imágenes LR con manejo de errores
    def load_and_process_lr_image(file_path):
        try:
            img = tf.io.read_file(file_path)
            # Intentar decodificar como JPEG, si falla, intentar como PNG
            try:
                img = tf.image.decode_jpeg(img, channels=3)
            except tf.errors.InvalidArgumentError:
                try:
                    img = tf.image.decode_png(img, channels=3)
                except:
                    logging.error(f"Error al decodificar imagen: {file_path}")
                    # Crear una imagen en negro como fallback
                    img = tf.zeros((*lr_size, 3), dtype=tf.uint8)
            
            # Convertir a float32 antes de redimensionar
            img = tf.cast(img, tf.float32)
            
            # Verificar que la imagen tiene al menos 3 canales
            if img.shape[-1] != 3:
                img = tf.image.grayscale_to_rgb(img)

            current_shape = tf.shape(img)[:2]
            target_shape = tf.constant(lr_size, dtype=tf.int32)
            need_resize = tf.reduce_any(tf.not_equal(current_shape, target_shape))

            img = tf.cond(
                need_resize,
                lambda: tf.image.resize(img, lr_size, method='bicubic'),
                lambda: img  # Sin redimensionamiento innecesario
            )
            
            # Clip values between 0-255
            return tf.clip_by_value(img, 0, 255)
        except Exception as e:
            logging.error(f"Error en imagen LR {file_path}: {str(e)}")
            return tf.zeros((*lr_size, 3), dtype=tf.float32)
    
    # Función para cargar y procesar imágenes HR
    def load_and_process_hr_image(file_path):
        try:
            img = tf.io.read_file(file_path)
            # Intentar decodificar como JPEG, si falla, intentar como PNG
            try:
                img = tf.image.decode_jpeg(img, channels=3)
            except tf.errors.InvalidArgumentError:
                try:
                    img = tf.image.decode_png(img, channels=3)
                except:
                    logging.error(f"Error al decodificar imagen: {file_path}")
                    # Crear una imagen en negro como fallback
                    img = tf.zeros((*hr_size, 3), dtype=tf.uint8)
            
            # Convertir a float32 antes de redimensionar
            img = tf.cast(img, tf.float32)
            
            # Verificar que la imagen tiene al menos 3 canales
            if img.shape[-1] != 3:
                img = tf.image.grayscale_to_rgb(img)

            current_shape = tf.shape(img)[:2]
            target_shape = tf.constant(hr_size, dtype=tf.int32)
            need_resize = tf.reduce_any(tf.not_equal(current_shape, target_shape))

            img = tf.cond(
                need_resize,
                lambda: tf.image.resize(img, hr_size, method='bicubic'),
                lambda: img  # Sin redimensionamiento innecesario
            )
            
            # Clip values between 0-255
            return tf.clip_by_value(img, 0, 255)
        except Exception as e:
            logging.error(f"Error en imagen HR {file_path}: {str(e)}")
            return tf.zeros((*hr_size, 3), dtype=tf.float32)
    
    # Aplicar las funciones de carga
    lr_ds = lr_ds.map(
        load_and_process_lr_image, 
        num_parallel_calls=num_parallel_calls,  # Usar 16 en lugar de AUTOTUNE
        deterministic=True
    )
    hr_ds = hr_ds.map(
        load_and_process_hr_image,
        num_parallel_calls=num_parallel_calls,  # Usar 16 en lugar de AUTOTUNE
        deterministic=True
    )
    # Combinar datasets
    dataset = tf.data.Dataset.zip((lr_ds, hr_ds))
    
    # Aplicar transformaciones adicionales si se proporcionan
    if transforms:
        dataset = dataset.map(
            transforms, 
            num_parallel_calls=num_parallel_calls,  # ✅ CAMBIAR
            deterministic=True
        )
    dataset = dataset.prefetch(buffer_size // 4)

    # Configurar dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    
    if repeat:
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Cambiado de experimental.AUTOTUNE a AUTOTUNE
    
    return dataset

