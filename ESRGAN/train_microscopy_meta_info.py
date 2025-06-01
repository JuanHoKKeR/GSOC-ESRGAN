import os
import argparse
import yaml
from absl import logging
import tensorflow as tf
from lib.dataset import load_dataset_from_meta_info
from lib import settings, train, model, utils

# Usar la nueva API de precisión mixta para TF 2.11
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    policy = mixed_precision.global_policy()
    print(f"Mixed precision activada: {policy}")
except Exception as e:
    print(f"Error al configurar mixed precision: {e}")
    print("Continuando sin mixed precision...")

# Comprobar que la GPU está disponible
physical_devices = tf.config.list_physical_devices('GPU')
print("Dispositivos GPU disponibles:", physical_devices)
if not physical_devices:
    print("ADVERTENCIA: No se detectó ninguna GPU. El entrenamiento será muy lento.")
else:
    print(f"Detectadas {len(physical_devices)} GPUs")
    # Habilitar crecimiento de memoria
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            tf.config.experimental.set_virtual_device_configuration(
                device,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=None)]
            )
        except Exception as e:
            print(f"Error al configurar memory growth: {e}")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("Wandb no está disponible. Continuando sin tracking.")

def configure_performance():
    """Configura TensorFlow para máximo rendimiento con tu hardware"""
    
    # ✅ USAR TODOS LOS 16 THREADS DEL CPU
    tf.config.threading.set_intra_op_parallelism_threads(16)  # Operaciones dentro de ops
    tf.config.threading.set_inter_op_parallelism_threads(8)   # Operaciones entre ops
    
    # ✅ CONFIGURAR GPU PARA MÁXIMO RENDIMIENTO
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
def update_config_batch_size(config_path, new_batch_size):
    """Actualiza el batch size en el archivo de configuración"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config["batch_size"] = new_batch_size
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logging.info(f"✅ Batch size actualizado a {new_batch_size} en {config_path}")
        return True
    except Exception as e:
        logging.error(f"❌ Error al actualizar configuración: {e}")
        return False

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
            use_wandb=False,
            config_path=None):
        
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
        # la inicialización del dataset, pero tomamos los demás parámetros.
        self.settings = settings
        self.model_dir = model_dir
        self.summary_writer = summary_writer
        self.summary_writer_2 = summary_writer_2
        self.strategy = strategy if strategy is not None else utils.SingleDeviceStrategy()
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.batch_size = self.settings["batch_size"]
        self.config_path = config_path
        
        # Obtener dimensiones de la configuración
        dataset_args = self.settings.get("dataset", {})
        self.hr_dimension = dataset_args.get("hr_dimension", 256)
        self.lr_dimension = dataset_args.get("lr_dimension", 128)

        self.hr_meta_file = hr_meta_file
        self.lr_meta_file = lr_meta_file
        self.base_path = base_path
        
        # Comprobar que los archivos existen
        for file_path in [hr_meta_file, lr_meta_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No se encuentra el archivo {file_path}")
            
        self._load_dataset()

    def _load_dataset(self):
        logging.info(f"Cargando dataset desde meta_info...")
        logging.info(f"HR meta: {self.hr_meta_file}")
        logging.info(f"LR meta: {self.lr_meta_file}")
        
        # Cargar el dataset desde los archivos meta_info
        try:
            dataset_iter = load_dataset_from_meta_info(
                hr_meta_file=self.hr_meta_file,
                lr_meta_file=self.lr_meta_file,
                base_path=self.base_path,
                batch_size=self.batch_size,
                shuffle=True,
                lr_size=(self.lr_dimension, self.lr_dimension),
                hr_size=(self.hr_dimension, self.hr_dimension),
                num_parallel_calls=16
            )
            
            # Verificar que el dataset está funcionando correctamente
            try:
                sample = next(iter(dataset_iter))
                logging.info(f"Muestra de dataset - LR shape: {sample[0].shape}, HR shape: {sample[1].shape}")
            except Exception as e:
                logging.error(f"Error al obtener muestra del dataset: {e}")
                raise
            
            self.dataset = iter(dataset_iter)
            logging.info("Dataset cargado correctamente")
        except Exception as e:
            logging.error(f"Error al cargar el dataset: {e}")
            raise


    def smart_train_gan(self, generator, discriminator):
        """Entrena la fase GAN con ajuste automático de batch size"""
        max_attempts = 3
        original_batch_size = self.batch_size
        
        for attempt in range(max_attempts):
            try:
                current_batch_size = original_batch_size // (2 ** attempt)
                
                if current_batch_size < 2:
                    logging.error("❌ Batch size demasiado pequeño, abortando")
                    raise RuntimeError("No se puede reducir más el batch size")
                
                if attempt > 0:
                    logging.info(f"🔄 Intento {attempt + 1}: Reduciendo batch size a {current_batch_size}")
                    
                    # Actualizar configuración
                    if self.config_path and update_config_batch_size(self.config_path, current_batch_size):
                        # Recargar configuración
                        self.settings = settings.Settings(self.config_path)
                        self.batch_size = current_batch_size
                        
                        # Recargar dataset con nuevo batch size
                        self._load_dataset()
                
                # Intentar entrenar
                logging.info(f"🚀 Iniciando fase 2 con batch size {current_batch_size}")
                self.train_gan(generator, discriminator)
                
                # Si llegamos aquí, fue exitoso
                if attempt > 0:
                    logging.info(f"✅ Fase 2 completada exitosamente con batch size {current_batch_size}")
                return
                
            # 🔧 CORRECCIÓN CRÍTICA: Mejorar detección de errores de memoria
            except Exception as e:
                error_msg = str(e).lower()
                error_type = type(e).__name__
                
                # Detectar errores de memoria por múltiples criterios
                is_memory_error = (
                    "out of memory" in error_msg or 
                    "resource exhausted" in error_msg or
                    "oom when allocating" in error_msg or
                    "resourceexhaustederror" in error_type.lower() or
                    "internal error" in error_msg
                )
                
                if is_memory_error:
                    logging.warning(f"⚠️  Error de memoria detectado en intento {attempt + 1}")
                    logging.warning(f"    Tipo de error: {error_type}")
                    logging.warning(f"    Mensaje: {str(e)[:200]}...")
                    
                    if attempt < max_attempts - 1:
                        logging.info("🔄 Reintentando con batch size menor...")
                        continue
                    else:
                        logging.error("❌ Agotados todos los intentos de reducir batch size")
                        raise
                else:
                    # Si no es error de memoria, re-lanzar inmediatamente
                    logging.error(f"❌ Error no relacionado con memoria: {error_type}")
                    logging.error(f"    Mensaje: {str(e)[:200]}...")
                    raise
        
        raise RuntimeError("No se pudo completar el entrenamiento de fase 2")

def create_discriminator(config, batch_size, kimianet_weights_override=None):
    """🆕 Crea el discriminador según la configuración"""
    
    # Obtener configuración del discriminador
    discriminator_config = config.get("discriminator", {})
    discriminator_type = discriminator_config.get("type", "densenet").lower()
    
    # Determinar ruta de pesos KimiaNet
    kimianet_weights_path = None
    if kimianet_weights_override:
        # Prioridad 1: Argumento de línea de comandos
        kimianet_weights_path = kimianet_weights_override
        logging.info(f"🎯 Usando pesos KimiaNet del argumento: {kimianet_weights_path}")
    elif discriminator_config.get("kimianet_weights"):
        # Prioridad 2: Config YAML
        kimianet_weights_path = discriminator_config.get("kimianet_weights")
        logging.info(f"🎯 Usando pesos KimiaNet del config: {kimianet_weights_path}")
    
    # Verificar si el archivo existe
    if kimianet_weights_path and not os.path.exists(kimianet_weights_path):
        logging.warning(f"⚠️  No se encontró el archivo de pesos KimiaNet en {kimianet_weights_path}")
        kimianet_weights_path = None
    
    # Crear el discriminador según el tipo
    if discriminator_type == "densenet":
        logging.info("🏗️  Creando DenseNetDiscriminator con KimiaNet")
        return model.DenseNetDiscriminator(kimianet_weights_path=kimianet_weights_path)
    
    elif discriminator_type == "vgg":
        num_features = discriminator_config.get("num_features", 64)
        logging.info(f"🏗️  Creando VGGArch con {num_features} features")
        return model.VGGArch(batch_size=batch_size, num_features=num_features)
    
    elif discriminator_type == "optimized_vgg":
        num_features = discriminator_config.get("num_features", 32)
        logging.info(f"🏗️  Creando OptimizedVGGArch con {num_features} features")
        return model.OptimizedVGGArch(batch_size=batch_size, num_features=num_features)
    
    else:
        logging.error(f"❌ Tipo de discriminador desconocido: {discriminator_type}")
        raise ValueError(f"Tipo de discriminador '{discriminator_type}' no soportado. Usa: 'densenet', 'vgg', o 'optimized_vgg'")


def main():

    configure_performance() 

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
        default=None,
        help="Tamaño de batch inicial (si no se especifica, se usa el del config)")
    parser.add_argument(
        "--kimianet_weights",
        default=None,
        help="Ruta a los pesos de KimiaNet (opcional)")
    parser.add_argument(
        "--pretrained_model",
        default=None,
        help="Ruta al modelo ESRGAN preentrenado para inicializar el generador")
    parser.add_argument(
        "--wandb_project",
        default="esrgan-microscopy",
        help="Nombre del proyecto en wandb (default: esrgan-microscopy-68to124)")
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
    
    # Cargar y actualizar configuración
    if not os.path.exists(args.config):
        logging.error(f"Archivo de configuración {args.config} no encontrado")
        return
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.batch_size:
        config["batch_size"] = args.batch_size
        logging.info(f"Usando batch size del argumento: {args.batch_size}")
    else:
        logging.info(f"Usando batch size del config: {config.get('batch_size', 16)}")
    
    # Guardar configuración actualizada
    config_dir = os.path.dirname(args.config)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
        
    with open(args.config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Cargar configuración
    sett = settings.Settings(args.config)

    dataset_config = sett.get("dataset", {})
    lr_dim = dataset_config.get("lr_dimension", 128)
    hr_dim = dataset_config.get("hr_dimension", 256)
    
    logging.info(f"🎯 Configuración automática:")
    logging.info(f"   - LR dimension: {lr_dim}x{lr_dim}")
    logging.info(f"   - HR dimension: {hr_dim}x{hr_dim}")
    logging.info(f"   - Batch size inicial: {sett['batch_size']}")

    # Configurar wandb
    use_wandb = False
    if not args.no_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config={
                    "batch_size": sett["batch_size"],
                    "lr_dimension": lr_dim,
                    "hr_dimension": hr_dim,
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
    
    # Configurar strategy para GPU
    strategy = utils.SingleDeviceStrategy()
    
    # Crear writers de TensorBoard
    summary_writer_1 = tf.summary.create_file_writer(os.path.join(args.log_dir, "phase1"))
    summary_writer_2 = tf.summary.create_file_writer(os.path.join(args.log_dir, "phase2"))
    
    # Inicializar modelos
    logging.info("Inicializando modelos...")
    try:
        # Usar ruta de pesos KimiaNet si se proporciona, sino None
        discriminator = create_discriminator(
            config=config,
            batch_size=sett["batch_size"],
            kimianet_weights_override=args.kimianet_weights
        )

        #kimianet_weights_path = args.kimianet_weights
        #if kimianet_weights_path and not os.path.exists(kimianet_weights_path):
        #    logging.warning(f"No se encontró el archivo de pesos KimiaNet en {kimianet_weights_path}. Se usará sin pesos preentrenados.")
        #    kimianet_weights_path = None
            
        #discriminator = model.DenseNetDiscriminator(kimianet_weights_path=kimianet_weights_path)
        #discriminator = model.VGGArch(batch_size=sett["batch_size"], num_features=64)
        #discriminator = model.OptimizedVGGArch(batch_size=sett["batch_size"], num_features=32)
        generator = model.RRDBNet(out_channel=3)
        
        # Inicializar los parámetros del modelo - usar método call compatible con TF 2.11
        logging.info("Inicializando parámetros del generador...")
        
        dummy_input_shape = [1, lr_dim, lr_dim, 3]  # Ajuste acorde a lr_dim del config
        generator(tf.random.normal(dummy_input_shape), training=True)  #Ajustar acorde a las dimensiones de entrada esperadas
        logging.info(f"Generador inicializado con entrada {dummy_input_shape}")

        # Cargar pesos preentrenados si se proporcionó la ruta
        if args.pretrained_model and os.path.exists(args.pretrained_model):
            logging.info(f"Cargando modelo preentrenado desde: {args.pretrained_model}")
            generator = utils.load_pretrained_generator(generator, args.pretrained_model)
            logging.info("Modelo preentrenado cargado como base para fine-tuning")
        else:
            logging.info("No se proporcionó modelo preentrenado. Usando inicialización aleatoria.")


        
        logging.info("Modelos inicializados correctamente")
    except Exception as e:
        logging.error(f"Error al inicializar los modelos: {e}")
        raise
    
    # Inicializar trainer personalizado
    logging.info("Creando trainer customizado...")
    try:
        trainer = CustomTrainer(
            summary_writer=summary_writer_1,
            summary_writer_2=summary_writer_2,
            settings=sett,
            model_dir=args.model_dir,
            hr_meta_file=args.hr_meta_file,
            lr_meta_file=args.lr_meta_file,
            base_path=args.base_path,
            strategy=strategy,
            use_wandb=use_wandb,
            config_path=args.config
        )


        logging.info("Trainer creado correctamente")
    except Exception as e:
        logging.error(f"Error al crear el trainer: {e}")
        raise
    
    # Inicializar Stats
    stats_file = os.path.join(os.path.dirname(args.config), "stats.yaml")
    stats = settings.Stats(stats_file)
    
    # Entrenar según las fases especificadas
    phases = args.phase.lower().split("_")
    if "phase1" in phases:
        logging.info("Iniciando fase 1 (PSNR)")
        try:
            trainer.warmup_generator(generator)
            stats["train_step_1"] = True
            logging.info("Fase 1 completada")
        except Exception as e:
            logging.error(f"Error en fase 1: {e}")
            raise
    
    if "phase2" in phases:
        logging.info("Iniciando fase 2 (GAN)")
        try:
            trainer.smart_train_gan(generator, discriminator)
            #trainer.train_gan(generator, discriminator)
            stats["train_step_2"] = True
            logging.info("Fase 2 completada")
        except Exception as e:
            logging.error(f"Error en fase 2: {e}")
            raise
    
    # Guardar modelo interpolado si se completaron ambas fases
    if stats["train_step_1"] and stats["train_step_2"]:
        logging.info("Guardando modelo interpolado")
        try:
            # Obtener dimensiones de la configuración o usar valores predeterminados
            # = sett.get("dataset", {}).get("hr_dimension", 256)
            #lr_dim = sett.get("dataset", {}).get("lr_dimension", 128)
            # Usar dimensions proporcionadas por la configuración
            interp_param = sett.get("interpolation_parameter", 0.8)
            
            interpolated_generator = utils.interpolate_generator(
                lambda: model.RRDBNet(out_channel=3, first_call=False),
                discriminator,
                interp_param,
                [hr_dim, hr_dim],  # Usar dimensiones consistentes
                basepath=args.model_dir
            )
            
            # Guardar el modelo en formato SavedModel
            save_path = os.path.join(args.model_dir, "esrgan")
            try:
                tf.saved_model.save(interpolated_generator, save_path)
                logging.info(f"Modelo guardado en {save_path}")
            except Exception as e:
                logging.error(f"Error al guardar el modelo en formato SavedModel: {e}")
                
                # Intento alternativo: guardar los pesos
                try:
                    weights_path = os.path.join(args.model_dir, "esrgan_weights")
                    interpolated_generator.save_weights(weights_path)
                    logging.info(f"Pesos del modelo guardados en {weights_path}")
                except Exception as e2:
                    logging.error(f"Error al guardar los pesos del modelo: {e2}")
            
            # Guardar modelo en wandb
            if use_wandb:
                try:
                    wandb.save(os.path.join(args.model_dir, "esrgan*"))
                    wandb.finish()
                    logging.info("Modelo registrado en wandb")
                except Exception as e:
                    logging.error(f"Error al guardar modelo en wandb: {e}")
                
            logging.info("Modelo guardado correctamente")
        except Exception as e:
            logging.error(f"Error al guardar modelo: {e}")
            raise
    
    logging.info("Entrenamiento completado")

if __name__ == "__main__":
    # Configurar manejo de excepciones global
    try:
        main()
    except Exception as e:
        logging.error(f"Error fatal: {e}", exc_info=True)
        # Si wandb está activo, finalizarlo correctamente
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish(exit_code=1)
        raise