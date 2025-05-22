import os
import time
from functools import partial
from absl import logging
import tensorflow as tf
from lib import utils, dataset
from tensorflow.keras.optimizers.legacy import Adam
try:
  import wandb
  WANDB_AVAILABLE = True
except ImportError:
  WANDB_AVAILABLE = False


class Trainer(object):
  """ Trainer class for ESRGAN """

  def __init__(
          self,
          summary_writer,
          summary_writer_2,
          settings,
          model_dir="",
          data_dir=None,
          manual=False,
          strategy=None,
          use_wandb=False):  # Añadido parámetro para usar wandb
    """ Setup the values and variables for Training.
        Args:
          summary_writer: tf.summary.SummaryWriter object to write summaries for Tensorboard.
          summary_writer_2: tf.summary.SummaryWriter object for phase 2 summaries.
          settings: settings object for fetching data from config files.
          model_dir (default: ""): path where the model checkpoints should be stored.
          data_dir (default: None): path where the data downloaded should be stored / accessed.
          manual (default: False): boolean to represent if data_dir is a manual dir.
          strategy (default: None): tf.distribute.Strategy object for distributed training.
          use_wandb (default: False): boolean to indicate if wandb logging is enabled.
    """
    self.settings = settings
    self.model_dir = model_dir
    self.summary_writer = summary_writer
    self.summary_writer_2 = summary_writer_2
    self.strategy = strategy
    self.use_wandb = use_wandb and WANDB_AVAILABLE  # Inicializar uso de wandb
    
    dataset_args = self.settings["dataset"]
    augment_dataset = dataset.augment_image(saturation=None)
    self.batch_size = self.settings["batch_size"]
    hr_size = tf.convert_to_tensor(
        [dataset_args["hr_dimension"],
         dataset_args["hr_dimension"], 3])
    lr_size = tf.cast(hr_size, tf.float32) * \
        tf.convert_to_tensor([1 / 4, 1 / 4, 1], tf.float32)
    lr_size = tf.cast(lr_size, tf.int32)
    
    # Configurar el dataset según la estrategia
    if isinstance(strategy, tf.distribute.Strategy):
      self.dataset = (dataset.load_tfrecord_dataset(
          tfrecord_path=data_dir,
          lr_size=lr_size,
          hr_size=hr_size,
          batch_size=self.batch_size)  # Pasar batch_size para usar con la función actualizada
          .repeat()
          .map(augment_dataset, num_parallel_calls=tf.data.AUTOTUNE))  # Usar AUTOTUNE para mejor rendimiento
          
      # En TF 2.11, preferir distribute_dataset en lugar de experimental_distribute_dataset
      if hasattr(strategy, 'distribute_dataset'):
          self.dataset = strategy.distribute_dataset(self.dataset)
      else:
          self.dataset = strategy.experimental_distribute_dataset(self.dataset)
          
      self.dataset = iter(self.dataset)
    else:
      # Inicializar self.dataset
      if not manual:
        scale_fn = dataset.scale_down(
                method=dataset_args["scale_method"],
                dimension=dataset_args["hr_dimension"])
        
        self.dataset = dataset.load_dataset(
            dataset_args["name"],
            scale_fn,
            batch_size=settings["batch_size"],
            data_dir=data_dir,
            augment=True,
            shuffle=True)
        self.dataset = self.dataset.repeat()
        self.dataset = iter(self.dataset)
      else:
        scale_fn = dataset.scale_down(
                method=dataset_args["scale_method"],
                dimension=dataset_args["hr_dimension"])
                
        self.dataset = dataset.load_dataset_directory(
            dataset_args["name"],
            data_dir,
            scale_fn,
            batch_size=settings["batch_size"],
            augment=True,
            shuffle=True)
        self.dataset = self.dataset.repeat()
        self.dataset = iter(self.dataset)

  def warmup_generator(self, generator):
    """ Training on L1 Loss to warmup the Generator.

    Minimizing the L1 Loss will reduce the Peak Signal to Noise Ratio (PSNR)
    of the generated image from the generator.
    This trained generator is then used to bootstrap the training of the
    GAN, creating better image inputs instead of random noises.
    Args:
      generator: Model Object for the Generator
    """
    # Loading up phase parameters
    warmup_num_iter = self.settings.get("warmup_num_iter", None)
    phase_args = self.settings["train_psnr"]
    decay_params = phase_args["adam"]["decay"]
    decay_step = decay_params["step"]
    decay_factor = decay_params["factor"]
    total_steps = phase_args["num_steps"]
    metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()
    # Métricas adicionales
    ssim_metric = tf.keras.metrics.Mean()
    ms_ssim_metric = tf.keras.metrics.Mean()
    mse_metric = tf.keras.metrics.Mean()
    # Generator Optimizer
    G_optimizer = Adam(
        learning_rate=phase_args["adam"]["initial_lr"],
        beta_1=phase_args["adam"]["beta_1"],
        beta_2=phase_args["adam"]["beta_2"])
    checkpoint = tf.train.Checkpoint(
        G=generator,
        G_optimizer=G_optimizer)

    status = utils.load_checkpoint(checkpoint, "phase_1", self.model_dir)
    logging.debug("phase_1 status object: {}".format(status))
    previous_loss = 0
    start_time = time.time()
    # Training starts

    def _step_fn(**kwargs):
      image_lr = kwargs["image_lr"]
      image_hr = kwargs["image_hr"]
      logging.debug("Starting Distributed Step")
      with tf.GradientTape() as tape:
        # Usar generator.call con training=True para asegurar que el modelo esté en modo entrenamiento
        if hasattr(generator, "unsigned_call"):
            fake = generator.unsigned_call(image_lr)
        else:
            fake = generator(image_lr, training=True)

        loss = utils.pixel_loss(image_hr, fake) * (1.0 / self.batch_size)
      
      # Calcular métricas
      psnr_metric.update_state(
          tf.reduce_mean(
              tf.image.psnr(
                  fake,
                  image_hr,
                  max_val=255.0)))  # Usar 255.0 en lugar de 256.0 para max_val
      
      # Actualizar métricas adicionales
      ssim_metric.update_state(utils.calculate_ssim(fake, image_hr))
      ms_ssim_metric.update_state(utils.calculate_ms_ssim(fake, image_hr))
      mse_metric.update_state(utils.calculate_mse(fake, image_hr))
      
      # Usar variables trainable directamente sin convertir a set
      gen_vars = generator.trainable_variables
      gradient = tape.gradient(loss, gen_vars)
      G_optimizer.apply_gradients(zip(gradient, gen_vars))
      
      # Actualizar métrica principal
      metric.update_state(loss)
      
      logging.debug("Ending Distributed Step")
      return tf.cast(G_optimizer.iterations, tf.float32)

    @tf.function
    def train_step(image_lr, image_hr):
        # Uso de run en Strategy para ejecutar el paso de entrenamiento
        if isinstance(self.strategy, tf.distribute.Strategy):
            distributed_metric = self.strategy.run(
                _step_fn, kwargs={"image_lr": image_lr, "image_hr": image_hr})
            mean_metric = self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, distributed_metric, axis=None)
            return mean_metric
        else:
            # Si no hay estrategia, ejecutar directamente
            return _step_fn(image_lr=image_lr, image_hr=image_hr)

    while True:
      image_lr, image_hr = next(self.dataset)
      num_steps = train_step(image_lr, image_hr)
      
      # Verificar si hemos alcanzado el número total de pasos
      if isinstance(num_steps, tf.Tensor):
          current_step = int(num_steps.numpy())
      else:
          current_step = int(num_steps)
          
      if current_step >= total_steps:
        return
      
      # Verificar el estado del checkpoint
      if status:
        try:
            status.assert_consumed()
            logging.info("Consumed checkpoint for phase_1 successfully")
            status = None
        except Exception as e:
            logging.warning(f"Error al consumir checkpoint: {e}")
            status = None

      # Ajustar el learning rate según el decay
      if current_step > 0 and current_step % decay_step == 0:
        # Acceder correctamente al learning rate
        current_lr = G_optimizer.learning_rate
        if hasattr(current_lr, 'numpy'):
            # Para TensorFlow 2.11, learning_rate puede ser un Tensor
            current_lr_value = float(current_lr.numpy())
        else:
            # Para compatibilidad con versiones anteriores, podría ser un valor escalar
            current_lr_value = float(current_lr)
            
        logging.debug(f"Current Learning Rate: {current_lr_value}")
        
        # Asignar el nuevo learning rate
        G_optimizer.learning_rate.assign(current_lr * decay_factor)
        
        logging.debug(
            f"Decayed Learning Rate by {decay_factor}. "
            f"New Learning Rate: {float(G_optimizer.learning_rate.numpy())}")
            
      # Escribir métricas en TensorBoard
      with self.summary_writer.as_default():
        tf.summary.scalar(
            "warmup_loss", metric.result(), step=current_step)
        tf.summary.scalar("mean_psnr", psnr_metric.result(), step=current_step)
        
        # Registrar métricas adicionales
        tf.summary.scalar("mean_ssim", ssim_metric.result(), step=current_step)
        tf.summary.scalar("mean_mse", mse_metric.result(), step=current_step)
        tf.summary.scalar("mean_ms_ssim", ms_ssim_metric.result(), step=current_step)
        
      # Registrar métricas en wandb si está habilitado
      if self.use_wandb and current_step % self.settings["print_step"] == 0:
        # Convertir a valores Python para evitar problemas con tensores
        try:
            loss_value = float(metric.result().numpy())
            psnr_value = float(psnr_metric.result().numpy())
            ssim_value = float(ssim_metric.result().numpy())
            ms_ssim_value = float(ms_ssim_metric.result().numpy())
            mse_value = float(mse_metric.result().numpy())
            current_lr = float(G_optimizer.learning_rate.numpy())
            
            # Registrar en wandb
            wandb.log({
                "phase1/loss": loss_value,
                "phase1/psnr": psnr_value,
                "phase1/ssim": ssim_value,
                "phase1/ms_ssim": ms_ssim_value,
                "phase1/mse": mse_value,
                "phase1/learning_rate": current_lr,
                "phase1/step": current_step,
            })
            
            # Registrar imágenes cada cierto número de pasos
            if current_step % (self.settings["print_step"] * 10) == 0:
                # Seleccionar un ejemplo del batch para visualización
                if isinstance(image_lr, tf.distribute.DistributedValues):
                    # Si estamos en entrenamiento distribuido, obtener el primer valor
                    sample_lr = image_lr.values[0][0]
                    sample_hr = image_hr.values[0][0]
                else:
                    sample_lr = image_lr[0]
                    sample_hr = image_hr[0]
                
                # Generar imagen usando el generador
                sample_fake = generator(tf.expand_dims(sample_lr, 0), training=False)[0]
                
                # Convertir a formato adecuado para wandb
                sample_lr = tf.cast(tf.clip_by_value(sample_lr, 0, 255), tf.uint8).numpy()
                sample_hr = tf.cast(tf.clip_by_value(sample_hr, 0, 255), tf.uint8).numpy()
                sample_fake = tf.cast(tf.clip_by_value(sample_fake, 0, 255), tf.uint8).numpy()
                
                # Registrar imágenes
                wandb.log({
                    "phase1/samples": [
                        wandb.Image(sample_lr, caption="Low Resolution"),
                        wandb.Image(sample_fake, caption="Generated"),
                        wandb.Image(sample_hr, caption="High Resolution")
                    ],
                    "phase1/step": current_step,
                })
        except Exception as e:
            logging.warning(f"Error al registrar métricas en wandb: {e}")

      # Imprimir información cada cierto número de pasos
      if current_step % self.settings["print_step"] == 0:
        logging.info(
            "[WARMUP] Step: {}\tGenerator Loss: {:.6f}"
            "\tPSNR: {:.4f} \tSSIM: {:.4f} \tMS-SSIM: {:.4f} \tMSE: {:.6f} \tTime Taken: {:.2f} sec".format(
                current_step,
                float(metric.result().numpy()),
                float(psnr_metric.result().numpy()),
                float(ssim_metric.result().numpy()),
                float(ms_ssim_metric.result().numpy()),
                float(mse_metric.result().numpy()),
                time.time() - start_time))
                
        # Guardar checkpoint si el PSNR mejora
        if psnr_metric.result() > previous_loss:
          utils.save_checkpoint(checkpoint, "phase_1", self.model_dir)
          previous_loss = psnr_metric.result()
          
        # Reiniciar el temporizador
        start_time = time.time()
        
        # Reiniciar las métricas para el próximo conjunto de pasos
        metric.reset_states()
        psnr_metric.reset_states()
        ssim_metric.reset_states()
        ms_ssim_metric.reset_states()
        mse_metric.reset_states()

  def train_gan(self, generator, discriminator):
    """ Implements Training routine for ESRGAN
        Args:
          generator: Model object for the Generator
          discriminator: Model object for the Discriminator
    """
    phase_args = self.settings["train_combined"]
    decay_args = phase_args["adam"]["decay"]
    decay_factor = decay_args["factor"]
    decay_steps = decay_args["step"].copy()  # Crear una copia para evitar modificar el original
    lambda_ = phase_args["lambda"]
    hr_dimension = self.settings["dataset"]["hr_dimension"]
    eta = phase_args["eta"]
    total_steps = phase_args["num_steps"]
    
    optimizer = partial(
        Adam,
        learning_rate=phase_args["adam"]["initial_lr"],
        beta_1=phase_args["adam"]["beta_1"],
        beta_2=phase_args["adam"]["beta_2"])

    G_optimizer = optimizer()
    D_optimizer = optimizer()

    ra_gen = utils.RelativisticAverageLoss(discriminator, type_="G")
    ra_disc = utils.RelativisticAverageLoss(discriminator, type_="D")

    # The weights of generator trained during Phase #1
    # is used to initialize or "hot start" the generator
    # for phase #2 of training
    status = None
    checkpoint = tf.train.Checkpoint(
        G=generator,
        G_optimizer=G_optimizer,
        D=discriminator,
        D_optimizer=D_optimizer)
        
    # Verificar si existe un checkpoint de fase 2
    checkpoint_dir = os.path.join(
        self.model_dir,
        self.settings["checkpoint_path"]["phase_2"])
    
    if not tf.io.gfile.exists(os.path.join(checkpoint_dir, "checkpoint")):
        # Si no existe checkpoint de fase 2, cargar desde fase 1
        hot_start = tf.train.Checkpoint(
            G=generator,
            G_optimizer=G_optimizer)
        status = utils.load_checkpoint(hot_start, "phase_1", self.model_dir)
        # Resetear el learning rate
        G_optimizer.learning_rate.assign(phase_args["adam"]["initial_lr"])
    else:
        # Si existe, cargar desde fase 2
        status = utils.load_checkpoint(checkpoint, "phase_2", self.model_dir)

    logging.debug("Phase 2 status object: {}".format(status))

    # Inicializar métricas
    gen_metric = tf.keras.metrics.Mean()
    disc_metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()
    
    # Métricas adicionales
    ssim_metric = tf.keras.metrics.Mean()
    ms_ssim_metric = tf.keras.metrics.Mean()
    mse_metric = tf.keras.metrics.Mean()
    
    logging.debug("Loading Perceptual Model")
    perceptual_loss = utils.PerceptualLoss(
        weights="imagenet",
        input_shape=[hr_dimension, hr_dimension, 3],
        loss_type=phase_args["perceptual_loss_type"])
    logging.debug("Loaded Perceptual Model")
    
    def _step_fn(**kwargs):
      image_lr = kwargs["image_lr"]
      image_hr = kwargs["image_hr"]
      logging.debug("Starting Distributed Step")
      
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Usar call con training=True

        if hasattr(generator, "unsigned_call"):
            fake = generator.unsigned_call(image_lr)
        else:
            fake = generator(image_lr, training=True)
        
        logging.debug("Generated fake image")
        
        # Pre-procesar imágenes si es necesario
        fake_processed = utils.preprocess_input(fake)
        lr_processed = utils.preprocess_input(image_lr)
        hr_processed = utils.preprocess_input(image_hr)
        
        # Calcular pérdidas
        percep_loss = tf.reduce_mean(perceptual_loss(hr_processed, fake_processed))
        logging.debug("Calculated Perceptual Loss")
        
        l1_loss = utils.pixel_loss(hr_processed, fake_processed)
        logging.debug("Calculated Pixel Loss")
        
        # Calcular pérdidas relativistas
        loss_RaG = ra_gen(hr_processed, fake_processed)
        logging.debug("Calculated Relativistic Average Loss for Generator")
        
        disc_loss = ra_disc(hr_processed, fake_processed)
        logging.debug("Calculated RA Loss Discriminator")
        
        # Pérdida combinada del generador
        gen_loss = percep_loss + lambda_ * loss_RaG + eta * l1_loss
        logging.debug("Calculated Generator Loss")
        
        # Escalar pérdidas por tamaño de batch
        gen_loss = gen_loss * (1.0 / self.batch_size)
        disc_loss = disc_loss * (1.0 / self.batch_size)
        
        # Actualizar métricas
        disc_metric.update_state(disc_loss)
        gen_metric.update_state(gen_loss)
        
        # Calcular PSNR y otras métricas
        psnr_metric.update_state(
            tf.reduce_mean(
                tf.image.psnr(
                    fake_processed,
                    hr_processed,
                    max_val=255.0)))  # Usar 255.0 para max_val
        
        # Actualizar métricas adicionales
        ssim_metric.update_state(utils.calculate_ssim(fake_processed, hr_processed))
        ms_ssim_metric.update_state(utils.calculate_ms_ssim(fake_processed, hr_processed))
        mse_metric.update_state(utils.calculate_mse(fake_processed, hr_processed))
        
      # Calcular y aplicar gradientes del discriminador
      disc_vars = discriminator.trainable_variables
      disc_grad = disc_tape.gradient(disc_loss, disc_vars)
      logging.debug("Calculated gradient for Discriminator")
      D_optimizer.apply_gradients(zip(disc_grad, disc_vars))
      logging.debug("Applied gradients to Discriminator")
      
      # Calcular y aplicar gradientes del generador
      gen_vars = generator.trainable_variables
      gen_grad = gen_tape.gradient(gen_loss, gen_vars)
      logging.debug("Calculated gradient for Generator")
      G_optimizer.apply_gradients(zip(gen_grad, gen_vars))
      logging.debug("Applied gradients to Generator")

      return tf.cast(D_optimizer.iterations, tf.float32)

    @tf.function
    def train_step(image_lr, image_hr):
        # Usar la estrategia para ejecutar el paso de entrenamiento
        if isinstance(self.strategy, tf.distribute.Strategy):
            distributed_iterations = self.strategy.run(
                _step_fn, kwargs={"image_lr": image_lr, "image_hr": image_hr})
            num_steps = self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                distributed_iterations,
                axis=None)
            return num_steps
        else:
            # Si no hay estrategia, ejecutar directamente
            return _step_fn(image_lr=image_lr, image_hr=image_hr)
            
    start = time.time()
    last_psnr = 0
    
    while True:
      image_lr, image_hr = next(self.dataset)
      num_step = train_step(image_lr, image_hr)
      
      # Convertir a valor Python si es un tensor
      if isinstance(num_step, tf.Tensor):
          current_step = int(num_step.numpy())
      else:
          current_step = int(num_step)
          
      # Verificar si hemos alcanzado el número total de pasos
      if current_step >= total_steps:
        return
        
      # Verificar el estado del checkpoint
      if status:
        try:
            status.assert_consumed()
            logging.info("Consumed checkpoint successfully!")
            status = None
        except Exception as e:
            logging.warning(f"Error al consumir checkpoint: {e}")
            status = None
            
      # Ajustar learning rate según decay_steps
      if decay_steps and current_step >= decay_steps[0]:
          # Eliminar el primer elemento de decay_steps
          decay_steps.pop(0)
          
          # Obtener y mostrar los learning rates actuales
          g_current_lr = G_optimizer.learning_rate
          d_current_lr = D_optimizer.learning_rate
          
          if isinstance(self.strategy, tf.distribute.Strategy):
              # Para entrenamiento distribuido
              g_current_lr = self.strategy.reduce(
                  tf.distribute.ReduceOp.MEAN,
                  g_current_lr, axis=None)
              d_current_lr = self.strategy.reduce(
                  tf.distribute.ReduceOp.MEAN,
                  d_current_lr, axis=None)
          
          logging.debug(
              f"Current LR: G = {float(g_current_lr.numpy())}, D = {float(d_current_lr.numpy())}")
          logging.debug(
              f"[Phase 2] Decayed Learning Rate by {decay_factor}")
              
          # Aplicar el decay a ambos optimizadores
          G_optimizer.learning_rate.assign(g_current_lr * decay_factor)
          D_optimizer.learning_rate.assign(d_current_lr * decay_factor)

      # Escribir métricas en TensorBoard
      with self.summary_writer_2.as_default():
        tf.summary.scalar(
            "gen_loss", gen_metric.result(), step=current_step)
        tf.summary.scalar(
            "disc_loss", disc_metric.result(), step=current_step)
        tf.summary.scalar("mean_psnr", psnr_metric.result(), step=current_step)
        
        # Registrar métricas adicionales
        tf.summary.scalar("mean_ssim", ssim_metric.result(), step=current_step)
        tf.summary.scalar("mean_mse", mse_metric.result(), step=current_step)
        tf.summary.scalar("mean_ms_ssim", ms_ssim_metric.result(), step=current_step)

      # Registrar métricas en wandb si está habilitado
      if self.use_wandb and current_step % self.settings["print_step"] == 0:
        try:
            # Convertir a valores Python para evitar problemas con tensores
            gen_loss_value = float(gen_metric.result().numpy())
            disc_loss_value = float(disc_metric.result().numpy())
            psnr_value = float(psnr_metric.result().numpy())
            ssim_value = float(ssim_metric.result().numpy())
            ms_ssim_value = float(ms_ssim_metric.result().numpy())
            mse_value = float(mse_metric.result().numpy())
            
            # Obtener learning rates actuales
            g_current_lr = float(G_optimizer.learning_rate.numpy())
            d_current_lr = float(D_optimizer.learning_rate.numpy())
            
            # Registrar en wandb
            wandb.log({
                "phase2/gen_loss": gen_loss_value,
                "phase2/disc_loss": disc_loss_value,
                "phase2/psnr": psnr_value,
                "phase2/ssim": ssim_value,
                "phase2/ms_ssim": ms_ssim_value,
                "phase2/mse": mse_value,
                "phase2/G_learning_rate": g_current_lr,
                "phase2/D_learning_rate": d_current_lr,
                "phase2/step": current_step,
            })
            
            # Registrar imágenes cada cierto número de pasos
            if current_step % (self.settings["print_step"] * 10) == 0:
                # Seleccionar un ejemplo para visualización
                if isinstance(image_lr, tf.distribute.DistributedValues):
                    # Si estamos en entrenamiento distribuido, obtener el primer valor
                    sample_lr = image_lr.values[0][0]
                    sample_hr = image_hr.values[0][0]
                else:
                    sample_lr = image_lr[0]
                    sample_hr = image_hr[0]
                
                # Generar imagen con el generador (sin entrenamiento)
                sample_fake = generator(tf.expand_dims(sample_lr, 0), training=False)[0]
                
                # Procesar las imágenes para visualización
                # Función para deshacer el preprocessing (ajustar según utils.preprocess_input)
                def undo_preprocess(img):
                    # Ejemplo básico - ajustar según la implementación real
                    mean = tf.constant([103.939, 116.779, 123.68])
                    return img[..., ::-1] + mean if hasattr(img, 'numpy') else img
                
                # Visualizar las imágenes
                sample_lr_vis = tf.cast(tf.clip_by_value(sample_lr, 0, 255), tf.uint8).numpy()
                sample_hr_vis = tf.cast(tf.clip_by_value(sample_hr, 0, 255), tf.uint8).numpy()
                sample_fake_vis = tf.cast(tf.clip_by_value(sample_fake, 0, 255), tf.uint8).numpy()
                
                # Registrar imágenes en wandb
                wandb.log({
                    "phase2/samples": [
                        wandb.Image(sample_lr_vis, caption="Low Resolution"),
                        wandb.Image(sample_fake_vis, caption="Generated"),
                        wandb.Image(sample_hr_vis, caption="High Resolution")
                    ],
                    "phase2/step": current_step,
                })
        except Exception as e:
            logging.warning(f"Error al registrar en wandb: {e}")

      # Imprimir información cada cierto número de pasos
      if current_step % self.settings["print_step"] == 0:
        # Formatear métricas para mostrar
        logging.info(
            "Step: {}\tGen Loss: {:.6f}\tDisc Loss: {:.6f}"
            "\tPSNR: {:.4f} \tSSIM: {:.4f} \tMS-SSIM: {:.4f} \tMSE: {:.6f} \tTime Taken: {:.2f} sec".format(
                current_step,
                float(gen_metric.result().numpy()),
                float(disc_metric.result().numpy()),
                float(psnr_metric.result().numpy()),
                float(ssim_metric.result().numpy()),
                float(ms_ssim_metric.result().numpy()),
                float(mse_metric.result().numpy()),
                time.time() - start))
                
        # Guardar checkpoint (siempre, ya que queremos el último estado)
        last_psnr = psnr_metric.result()
        utils.save_checkpoint(checkpoint, "phase_2", self.model_dir)
        
        # Reiniciar temporizador y métricas
        start = time.time()
        gen_metric.reset_states()
        disc_metric.reset_states()
        psnr_metric.reset_states()
        ssim_metric.reset_states()
        ms_ssim_metric.reset_states()
        mse_metric.reset_states()