import os
import time
from functools import partial
from absl import logging
import tensorflow as tf
from lib import utils, dataset
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
          strategy=None):
    """ Setup the values and variables for Training.
        Args:
          summary_writer: tf.summary.SummaryWriter object to write summaries for Tensorboard.
          settings: settings object for fetching data from config files.
          data_dir (default: None): path where the data downloaded should be stored / accessed.
          manual (default: False): boolean to represent if data_dir is a manual dir.
    """
    self.settings = settings
    self.model_dir = model_dir
    self.summary_writer = summary_writer
    self.summary_writer_2 = summary_writer_2
    self.strategy = strategy
    dataset_args = self.settings["dataset"]
    augment_dataset = dataset.augment_image(saturation=None)
    self.batch_size = self.settings["batch_size"]
    hr_size = tf.convert_to_tensor(
        [dataset_args["hr_dimension"],
         dataset_args["hr_dimension"], 3])
    lr_size = tf.cast(hr_size, tf.float32) * \
        tf.convert_to_tensor([1 / 4, 1 / 4, 1], tf.float32)
    lr_size = tf.cast(lr_size, tf.int32)
    if isinstance(strategy, tf.distribute.Strategy):
      self.dataset = (dataset.load_tfrecord_dataset(
          tfrecord_path=data_dir,
          lr_size=lr_size,
          hr_size=hr_size)
          .repeat()
          .map(augment_dataset)
          .batch(self.batch_size, drop_remainder=True))
      self.dataset = iter(
          strategy.experimental_distribute_dataset(
              self.dataset))
    else:
      if not manual:
        self.dataset.repeat()
        self.dataset = iter(dataset.load_dataset(
            dataset_args["name"],
            dataset.scale_down(
                method=dataset_args["scale_method"],
                dimension=dataset_args["hr_dimension"]),
            batch_size=settings["batch_size"],
            data_dir=data_dir,
            augment=True,
            shuffle=True))
      else:
        self.dataset.repeat()
        self.dataset = iter(dataset.load_dataset_directory(
            dataset_args["name"],
            data_dir,
            dataset.scale_down(
                method=dataset_args["scale_method"],
                dimension=dataset_args["hr_dimension"]),
            batch_size=settings["batch_size"],
            augment=True,
            shuffle=True))

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
    # Generator Optimizer
    G_optimizer = tf.optimizers.Adam(
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
        fake = generator.unsigned_call(image_lr)
        loss = utils.pixel_loss(image_hr, fake) * (1.0 / self.batch_size)
      psnr_metric(
          tf.reduce_mean(
              tf.image.psnr(
                  fake,
                  image_hr,
                  max_val=256.0)))
      # Eliminar la conversión a set que causa el error
      # gen_vars = list(set(generator.trainable_variables))
      gen_vars = generator.trainable_variables  # Usar directamente las variables
      gradient = tape.gradient(loss, gen_vars)
      G_optimizer.apply_gradients(
          zip(gradient, gen_vars))
      mean_loss = metric(loss)
      logging.debug("Ending Distributed Step")
      return tf.cast(G_optimizer.iterations, tf.float32)

    @tf.function
    def train_step(image_lr, image_hr):
        distributed_metric = self.strategy.run(
            _step_fn, args=[], kwargs={"image_lr": image_lr, "image_hr": image_hr})
        mean_metric = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, distributed_metric, axis=None)
        return mean_metric

    while True:
      image_lr, image_hr = next(self.dataset)
      num_steps = train_step(image_lr, image_hr)

      if num_steps >= total_steps:
        return
      if status:
        status.assert_consumed()
        logging.info(
            "consumed checkpoint for phase_1 successfully")
        status = None

      if not num_steps % decay_step:  # Decay Learning Rate
        logging.debug(
            "Learning Rate: %s" %
            G_optimizer.learning_rate.numpy)
        G_optimizer.learning_rate.assign(
            G_optimizer.learning_rate * decay_factor)
        logging.debug(
            "Decayed Learning Rate by %f."
            "Current Learning Rate %s" % (
                decay_factor, G_optimizer.learning_rate))
      with self.summary_writer.as_default():
        tf.summary.scalar(
            "warmup_loss", metric.result(), step=G_optimizer.iterations)
        tf.summary.scalar("mean_psnr", psnr_metric.result(), G_optimizer.iterations)

      # Registrar métricas en wandb si está habilitado
      if self.use_wandb and not num_steps % self.settings["print_step"]:
        # Convertir a numpy para evitar problemas con tensores de TensorFlow
        loss_value = float(metric.result().numpy())
        psnr_value = float(psnr_metric.result().numpy())
        current_lr = float(G_optimizer.learning_rate.numpy())
        
        # Registrar en wandb
        wandb.log({
            "phase1/loss": loss_value,
            "phase1/psnr": psnr_value,
            "phase1/learning_rate": current_lr,
            "phase1/step": int(num_steps.numpy()),
        })
        
        # Opcional: Añadir imagen generada como ejemplo cada cierto número de pasos
        if not num_steps % (self.settings["print_step"] * 10):
            # Selecciona un ejemplo de la batch
            sample_lr = image_lr[0]
            sample_hr = image_hr[0]
            sample_fake = generator.unsigned_call(tf.expand_dims(sample_lr, 0))[0]
            
            # Convertir a formato correcto para wandb
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
                "phase1/step": int(num_steps.numpy()),
            })      

      if not num_steps % self.settings["print_step"]:
        logging.info(
            "[WARMUP] Step: {}\tGenerator Loss: {}"
            "\tPSNR: {}\tTime Taken: {} sec".format(
                num_steps,
                metric.result(),
                psnr_metric.result(),
                time.time() -
                start_time))
        if psnr_metric.result() > previous_loss:
          utils.save_checkpoint(checkpoint, "phase_1", self.model_dir)
        previous_loss = psnr_metric.result()
        start_time = time.time()

  def train_gan(self, generator, discriminator):
    """ Implements Training routine for ESRGAN
        Args:
          generator: Model object for the Generator
          discriminator: Model object for the Discriminator
    """
    phase_args = self.settings["train_combined"]
    decay_args = phase_args["adam"]["decay"]
    decay_factor = decay_args["factor"]
    decay_steps = decay_args["step"]
    lambda_ = phase_args["lambda"]
    hr_dimension = self.settings["dataset"]["hr_dimension"]
    eta = phase_args["eta"]
    total_steps = phase_args["num_steps"]
    optimizer = partial(
        tf.optimizers.Adam,
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
    if not tf.io.gfile.exists(
        os.path.join(
            self.model_dir,
            self.settings["checkpoint_path"]["phase_2"],
            "checkpoint")):
      hot_start = tf.train.Checkpoint(
          G=generator,
          G_optimizer=G_optimizer)
      status = utils.load_checkpoint(hot_start, "phase_1", self.model_dir)
      # consuming variable from checkpoint
      G_optimizer.learning_rate.assign(phase_args["adam"]["initial_lr"])
    else:
      status = utils.load_checkpoint(checkpoint, "phase_2", self.model_dir)

    logging.debug("phase status object: {}".format(status))

    gen_metric = tf.keras.metrics.Mean()
    disc_metric = tf.keras.metrics.Mean()
    psnr_metric = tf.keras.metrics.Mean()
    logging.debug("Loading Perceptual Model")
    perceptual_loss = utils.PerceptualLoss(
        weights="imagenet",
        input_shape=[hr_dimension, hr_dimension, 3],
        loss_type=phase_args["perceptual_loss_type"])
    logging.debug("Loaded Model")
    def _step_fn(**kwargs):
      image_lr = kwargs["image_lr"]
      image_hr = kwargs["image_hr"]
      logging.debug("Starting Distributed Step")
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake = generator.unsigned_call(image_lr)
        logging.debug("Fetched Generator Fake")
        fake = utils.preprocess_input(fake)
        image_lr = utils.preprocess_input(image_lr)
        image_hr = utils.preprocess_input(image_hr)
        percep_loss = tf.reduce_mean(perceptual_loss(image_hr, fake))
        logging.debug("Calculated Perceptual Loss")
        l1_loss = utils.pixel_loss(image_hr, fake)
        logging.debug("Calculated Pixel Loss")
        loss_RaG = ra_gen(image_hr, fake)
        logging.debug("Calculated Relativistic"
                      "Averate (RA) Loss for Generator")
        disc_loss = ra_disc(image_hr, fake)
        logging.debug("Calculated RA Loss Discriminator")
        gen_loss = percep_loss + lambda_ * loss_RaG + eta * l1_loss
        logging.debug("Calculated Generator Loss")
        disc_metric(disc_loss)
        gen_metric(gen_loss)
        gen_loss = gen_loss * (1.0 / self.batch_size)
        disc_loss = disc_loss * (1.0 / self.batch_size)
        psnr_metric(
            tf.reduce_mean(
                tf.image.psnr(
                    fake,
                    image_hr,
                    max_val=256.0)))
      disc_grad = disc_tape.gradient(
          disc_loss, discriminator.trainable_variables)
      logging.debug("Calculated gradient for Discriminator")
      D_optimizer.apply_gradients(
          zip(disc_grad, discriminator.trainable_variables))
      logging.debug("Applied gradients to Discriminator")
      gen_grad = gen_tape.gradient(
          gen_loss, generator.trainable_variables)
      logging.debug("Calculated gradient for Generator")
      G_optimizer.apply_gradients(
          zip(gen_grad, generator.trainable_variables))
      logging.debug("Applied gradients to Generator")

      return tf.cast(D_optimizer.iterations, tf.float32)

    @tf.function
    def train_step(image_lr, image_hr):
        distributed_iterations = self.strategy.run(
            _step_fn, args=[], kwargs={"image_lr": image_lr, "image_hr": image_hr})
        num_steps = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            distributed_iterations,
            axis=None)
        return num_steps
    start = time.time()
    last_psnr = 0
    while True:
      image_lr, image_hr = next(self.dataset)
      num_step = train_step(image_lr, image_hr)
      if num_step >= total_steps:
        return
      if status:
        status.assert_consumed()
        logging.info("consumed checkpoint successfully!")
        status = None
      # Decaying Learning Rate
      for _step in decay_steps.copy():
        if num_step >= _step:
          decay_steps.pop(0)
          g_current_lr = self.strategy.reduce(
              tf.distribute.ReduceOp.MEAN,
              G_optimizer.learning_rate, axis=None)

          d_current_lr = self.strategy.reduce(
              tf.distribute.ReduceOp.MEAN,
              D_optimizer.learning_rate, axis=None)

          logging.debug(
              "Current LR: G = %s, D = %s" %
              (g_current_lr, d_current_lr))
          logging.debug(
              "[Phase 2] Decayed Learing Rate by %f." % decay_factor)
          G_optimizer.learning_rate.assign(
              G_optimizer.learning_rate * decay_factor)
          D_optimizer.learning_rate.assign(
              D_optimizer.learning_rate * decay_factor)

      # Writing Summary
      with self.summary_writer_2.as_default():
        tf.summary.scalar(
            "gen_loss", gen_metric.result(), step=D_optimizer.iterations)
        tf.summary.scalar(
            "disc_loss", disc_metric.result(), step=D_optimizer.iterations)
        tf.summary.scalar("mean_psnr", psnr_metric.result(), step=D_optimizer.iterations)

      # Registrar métricas en wandb si está habilitado
      if self.use_wandb and not num_step % self.settings["print_step"]:
        # Convertir a numpy para evitar problemas con tensores de TensorFlow
        gen_loss_value = float(gen_metric.result().numpy())
        disc_loss_value = float(disc_metric.result().numpy())
        psnr_value = float(psnr_metric.result().numpy())
        g_current_lr = float(G_optimizer.learning_rate.numpy())
        d_current_lr = float(D_optimizer.learning_rate.numpy())
        
        # Registrar en wandb
        wandb.log({
            "phase2/gen_loss": gen_loss_value,
            "phase2/disc_loss": disc_loss_value,
            "phase2/psnr": psnr_value,
            "phase2/G_learning_rate": g_current_lr,
            "phase2/D_learning_rate": d_current_lr,
            "phase2/step": int(num_step.numpy()),
        })
        
        # Opcional: Añadir imagen generada como ejemplo cada cierto número de pasos
        if not num_step % (self.settings["print_step"] * 10):
            # Selecciona un ejemplo de la batch
            sample_lr = image_lr[0]
            sample_hr = image_hr[0]
            sample_fake = generator.unsigned_call(tf.expand_dims(sample_lr, 0))[0]
            
            # Post-procesar las imágenes para visualización
            sample_lr = tf.cast(tf.clip_by_value(utils.preprocess_input(sample_lr) + tf.constant([103.939, 116.779, 123.68]), 0, 255), tf.uint8).numpy()
            sample_hr = tf.cast(tf.clip_by_value(utils.preprocess_input(sample_hr) + tf.constant([103.939, 116.779, 123.68]), 0, 255), tf.uint8).numpy()
            sample_fake = tf.cast(tf.clip_by_value(sample_fake, 0, 255), tf.uint8).numpy()
            
            # Registrar imágenes
            wandb.log({
                "phase2/samples": [
                    wandb.Image(sample_lr, caption="Low Resolution"),
                    wandb.Image(sample_fake, caption="Generated"),
                    wandb.Image(sample_hr, caption="High Resolution")
                ],
                "phase2/step": int(num_step.numpy()),
            })


      # Logging and Checkpointing
      if not num_step % self.settings["print_step"]:
        logging.info(
            "Step: {}\tGen Loss: {}\tDisc Loss: {}"
            "\tPSNR: {}\tTime Taken: {} sec".format(
                num_step,
                gen_metric.result(),
                disc_metric.result(),
                psnr_metric.result(),
                time.time() - start))
        # if psnr_metric.result() > last_psnr:
        last_psnr = psnr_metric.result()
        utils.save_checkpoint(checkpoint, "phase_2", self.model_dir)
        start = time.time()
