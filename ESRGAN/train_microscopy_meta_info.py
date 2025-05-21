import os
import argparse
import yaml
from absl import logging
import tensorflow as tf
from lib.dataset import load_dataset_from_meta_info
from lib import settings, train, model, utils


from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')

mixed_precision.set_policy(policy)
print("Mixed precision activada: ", policy)

# Comprobar que la GPU está disponible
physical_devices = tf.config.list_physical_devices('GPU')
print("Dispositivos GPU disponibles:", physical_devices)
if not physical_devices:
    print("ADVERTENCIA: No se detectó ninguna GPU. El entrenamiento será muy lento.")
else:
    print(f"Detectadas {len(physical_devices)} GPUs")
    # Habilitar crecimiento de memoria
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Wandb no está disponible. Continuando sin tracking.")

class CustomTrainer(train.Trainer):
    """Trainer personalizado para usar con archivos meta_info."""
    
    def __init__(
            self,
            summary_writer,
            summary_writer_2,
            settings,
            model_dir="",
            hr_meta_file=None,
            lr_meta_file=None,
            base_path="",
            strategy=None,
            use_wandb=False):
        """Inicializa el entrenador personalizado para dataset de microscopía.
        
        Args:
            summary_writer: Writer para TensorBoard fase 1
            summary_writer_2: Writer para TensorBoard fase 2
            settings: Objeto Settings con configuración
            model_dir: Directorio para guardar el modelo
            hr_meta_file: Ruta al archivo meta_info de alta resolución
            lr_meta_file: Ruta al archivo meta_info de baja resolución
            base_path: Ruta base para las imágenes
            strategy: Estrategia de distribución (None o SingleDeviceStrategy)
            use_wandb: Si se debe usar Weights & Biases
        """
        # No llamamos a super().__init__ porque queremos reemplazar completamente
        # la inicialización del dataset, pero tomamos los demás parámetros.
        self.settings = settings
        self.model_dir = model_dir
        self.summary_writer = summary_writer
        self.summary_writer_2 = summary_writer_2
        self.strategy = strategy if strategy is not None else utils.SingleDeviceStrategy()
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.batch_size = self.settings["batch_size"]
        
        # Comprobar que los archivos existen
        for file_path in [hr_meta_file, lr_meta_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No se encuentra el archivo {file_path}")
        
        logging.info(f"Cargando dataset desde meta_info...")
        logging.info(f"HR meta: {hr_meta_file}")
        logging.info(f"LR meta: {lr_meta_file}")
        
        # Cargar el dataset desde los archivos meta_info
        try:
            dataset_iter = load_dataset_from_meta_info(
                hr_meta_file=hr_meta_file,
                lr_meta_file=lr_meta_file,
                base_path=base_path,
                batch_size=self.batch_size,
                shuffle=True,
                lr_size=(128, 128),
                hr_size=(512, 512)
            )
            # Verificar que el dataset está funcionando correctamente
            sample = next(iter(dataset_iter))
            logging.info(f"Muestra de dataset - LR shape: {sample[0].shape}, HR shape: {sample[1].shape}")
            
            self.dataset = iter(dataset_iter)
            logging.info("Dataset cargado correctamente")
        except Exception as e:
            logging.error(f"Error al cargar el dataset: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Entrenar ESRGAN con imágenes de microscopía usando archivos meta_info y wandb")
    parser.add_argument(
        "--hr_meta_file",
        required=True,
        help="Archivo meta_info con rutas de imágenes de alta resolución")
    parser.add_argument(
        "--lr_meta_file",
        required=True,
        help="Archivo meta_info con rutas de imágenes de baja resolución")
    parser.add_argument(
        "--base_path",
        default="",
        help="Ruta base para las imágenes si las rutas en meta_info son relativas")
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directorio para guardar el modelo")
    parser.add_argument(
        "--log_dir",
        required=True,
        help="Directorio para guardar logs de TensorBoard")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al archivo de configuración")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Tamaño de batch (default: 4)")
    parser.add_argument(
        "--wandb_project",
        default="esrgan-microscopy",
        help="Nombre del proyecto en wandb (default: esrgan-microscopy)")
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="Entidad de wandb (opcional)")
    parser.add_argument(
        "--wandb_name",
        default=None,
        help="Nombre del experimento en wandb (opcional)")
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "phase1_phase2"],
        default="phase1_phase2",
        help="Fase de entrenamiento (default: phase1_phase2)")
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Deshabilitar Weight and Biases")
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Aumentar verbosidad")
    
    args = parser.parse_args()
    
    # Configurar nivel de log
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, len(log_levels) - 1)]
    logging.set_verbosity(log_level)
    
    # Configurar directorios
    for directory in [args.model_dir, args.log_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Configurar wandb
    use_wandb = False
    if not args.no_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config={
                    "batch_size": args.batch_size,
                    "hr_meta_file": args.hr_meta_file,
                    "lr_meta_file": args.lr_meta_file,
                    "phase": args.phase
                }
            )
            use_wandb = True
            logging.info("Weights & Biases inicializado correctamente")
        except Exception as e:
            logging.error(f"Error al inicializar Weights & Biases: {e}")
            logging.info("Continuando sin tracking de Weights & Biases")
    
    # Cargar y actualizar configuración
    if not os.path.exists(args.config):
        logging.error(f"Archivo de configuración {args.config} no encontrado")
        return
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Actualizar configuración con batch_size
    config["batch_size"] = args.batch_size
    
    # Guardar configuración actualizada
    with open(args.config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Cargar configuración
    sett = settings.Settings(args.config)
    
    # Configurar strategy para GPU
    strategy = utils.SingleDeviceStrategy()
    for physical_device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(physical_device, True)
    
    # Crear writers de TensorBoard
    summary_writer_1 = tf.summary.create_file_writer(os.path.join(args.log_dir, "phase1"))
    summary_writer_2 = tf.summary.create_file_writer(os.path.join(args.log_dir, "phase2"))
    
    # Inicializar modelos
    logging.info("Inicializando modelos...")
    discriminator = model.DenseNetDiscriminator(kimianet_weights_path="/model-kimianet/KimiaNetKerasWeights.h5")
    generator = model.RRDBNet(out_channel=3)
    
    # Inicializar los parámetros del modelo
    logging.info("Inicializando parámetros del generador...")
    generator.unsigned_call(tf.random.normal([1, 128, 128, 3]))
    
    # Inicializar trainer personalizado
    logging.info("Creando trainer customizado...")
    trainer = CustomTrainer(
        summary_writer=summary_writer_1,
        summary_writer_2=summary_writer_2,
        settings=sett,
        model_dir=args.model_dir,
        hr_meta_file=args.hr_meta_file,
        lr_meta_file=args.lr_meta_file,
        base_path=args.base_path,
        strategy=strategy,
        use_wandb=use_wandb
    )
    
    # Inicializar Stats
    stats_file = os.path.join(os.path.dirname(args.config), "stats.yaml")
    Stats = settings.Stats(stats_file)
    
    # Entrenar según las fases especificadas
    phases = args.phase.lower().split("_")
    if "phase1" in phases:
        logging.info("Iniciando fase 1 (PSNR)")
        try:
            trainer.warmup_generator(generator)
            Stats["train_step_1"] = True
            logging.info("Fase 1 completada")
        except Exception as e:
            logging.error(f"Error en fase 1: {e}")
            raise
    
    if "phase2" in phases:
        logging.info("Iniciando fase 2 (GAN)")
        try:
            trainer.train_gan(generator, discriminator)
            Stats["train_step_2"] = True
            logging.info("Fase 2 completada")
        except Exception as e:
            logging.error(f"Error en fase 2: {e}")
            raise
    
    # Guardar modelo interpolado si se completaron ambas fases
    if Stats["train_step_1"] and Stats["train_step_2"]:
        logging.info("Guardando modelo interpolado")
        try:
            interpolated_generator = utils.interpolate_generator(
                lambda: model.RRDBNet(out_channel=3, first_call=False),
                discriminator,
                sett["interpolation_parameter"],
                [720, 1080],
                basepath=args.model_dir
            )
            tf.saved_model.save(
                interpolated_generator, 
                os.path.join(args.model_dir, "esrgan")
            )
            
            # Guardar modelo en wandb
            if use_wandb:
                wandb.save(os.path.join(args.model_dir, "esrgan"))
                wandb.finish()
                
            logging.info("Modelo guardado correctamente")
        except Exception as e:
            logging.error(f"Error al guardar modelo: {e}")
            raise
    
    logging.info("Entrenamiento completado")

if __name__ == "__main__":
    main()