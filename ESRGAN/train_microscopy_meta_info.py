import os
import argparse
import yaml
from absl import logging
import wandb
import tensorflow as tf
from lib.dataset import load_dataset_from_meta_info
from lib import settings, train, model, utils

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
    
    # Configurar wandb
    if not args.no_wandb:
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
    
    # Cargar y actualizar configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Actualizar configuración
    config["batch_size"] = args.batch_size
    
    # Guardar configuración actualizada
    with open(args.config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Configurar directorios
    for directory in [args.model_dir, args.log_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Inicializar y entrenar modelo
    print("Iniciando entrenamiento con archivos meta_info:")
    print(f"HR meta_info: {args.hr_meta_file}")
    print(f"LR meta_info: {args.lr_meta_file}")
    
    # Cargar configuración
    sett = settings.Settings(args.config)
    
    # Configurar strategy
    strategy = utils.SingleDeviceStrategy()
    for physical_device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(physical_device, True)
    
    # Crear writers de TensorBoard
    summary_writer_1 = tf.summary.create_file_writer(os.path.join(args.log_dir, "phase1"))
    summary_writer_2 = tf.summary.create_file_writer(os.path.join(args.log_dir, "phase2"))
    
    # Inicializar modelos
    discriminator = model.VGGArch(batch_size=sett["batch_size"], num_features=64)
    generator = model.RRDBNet(out_channel=3)
    generator.unsigned_call(tf.random.normal([1, 128, 128, 3]))  # Inicializar generador
    
    # Definir la clase CustomTrainer que hereda de Trainer pero sobrescribe el método __init__
    class CustomTrainer(train.Trainer):
        def __init__(self, summary_writer, summary_writer_2, settings, model_dir="", 
                     hr_meta_file=None, lr_meta_file=None, base_path="", strategy=None, use_wandb=False):
            self.settings = settings
            self.model_dir = model_dir
            self.summary_writer = summary_writer
            self.summary_writer_2 = summary_writer_2
            self.strategy = strategy
            self.use_wandb = use_wandb
            self.batch_size = self.settings["batch_size"]
            
            # Cargar el dataset desde los archivos meta_info
            print(f"Cargando dataset desde meta_info...")
            self.dataset = iter(load_dataset_from_meta_info(
                hr_meta_file=hr_meta_file,
                lr_meta_file=lr_meta_file,
                base_path=base_path,
                batch_size=self.batch_size,  # Cambiado de 1000 a self.batch_size
                shuffle=True,
                lr_size=(128, 128),          # Añadir tamaño LR
                hr_size=(512, 512)           # Añadir tamaño HR
            ))
            print("Dataset cargado correctamente.")
    
    # Inicializar trainer personalizado
    trainer = CustomTrainer(
        summary_writer=summary_writer_1,
        summary_writer_2=summary_writer_2,
        settings=sett,
        model_dir=args.model_dir,
        hr_meta_file=args.hr_meta_file,
        lr_meta_file=args.lr_meta_file,
        base_path=args.base_path,
        strategy=strategy,
        use_wandb=not args.no_wandb
    )
    
    # Inicializar Stats
    Stats = settings.Stats(os.path.join(sett.path, "stats.yaml"))
    
    # Entrenar según las fases especificadas
    phases = args.phase.lower().split("_")
    if "phase1" in phases:
        logging.info("Iniciando fase 1 (PSNR)")
        trainer.warmup_generator(generator)
        Stats["train_step_1"] = True
    
    if "phase2" in phases:
        logging.info("Iniciando fase 2 (GAN)")
        trainer.train_gan(generator, discriminator)
        Stats["train_step_2"] = True
    
    # Guardar modelo interpolado si se completaron ambas fases
    if Stats["train_step_1"] and Stats["train_step_2"]:
        logging.info("Guardando modelo interpolado")
        interpolated_generator = utils.interpolate_generator(
            lambda: model.RRDBNet(out_channel=3, first_call=False),
            discriminator,
            sett["interpolation_parameter"],
            [720, 1080],
            basepath=args.model_dir
        )
        tf.saved_model.save(interpolated_generator, os.path.join(args.model_dir, "esrgan"))
        
        # Guardar modelo en wandb
        if not args.no_wandb:
            wandb.save(os.path.join(args.model_dir, "esrgan"))
            wandb.finish()
    
    logging.info("Entrenamiento completado")

if __name__ == "__main__":
    main()