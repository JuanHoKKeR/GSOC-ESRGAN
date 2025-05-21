import os
import yaml
from typing import Any, Dict, Optional


def singleton(cls):
    """Decorador para implementar el patrón Singleton.
    
    Asegura que solo exista una instancia de la clase decorada.
    
    Args:
        cls: Clase a convertir en singleton
        
    Returns:
        Función getinstance que devuelve la única instancia de la clase
    """
    instances = {}

    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return getinstance


@singleton
class Settings(object):
    """Clase para gestionar la configuración desde un archivo YAML.
    
    Esta clase implementa el patrón Singleton para asegurar
    que solo exista una instancia de configuración en toda la aplicación.
    """
    
    def __init__(self, filename="config.yaml"):
        """Inicializa el objeto Settings.
        
        Args:
            filename (str, optional): Ruta al archivo de configuración. 
                                     Por defecto es "config.yaml".
        """
        self.__path = os.path.abspath(filename)
        # Verificar que el archivo existe
        if not os.path.exists(self.__path):
            raise FileNotFoundError(f"El archivo de configuración {self.__path} no existe")

    @property
    def path(self) -> str:
        """Obtiene el directorio donde se encuentra el archivo de configuración.
        
        Returns:
            str: Ruta al directorio que contiene el archivo de configuración
        """
        return os.path.dirname(self.__path)

    def __getitem__(self, index: str) -> Any:
        """Accede a un elemento de configuración por su clave.
        
        Args:
            index (str): Clave del elemento de configuración a obtener
            
        Returns:
            Any: Valor asociado a la clave en el archivo de configuración
            
        Raises:
            KeyError: Si la clave no existe en el archivo de configuración
            yaml.YAMLError: Si hay un error en el formato del archivo YAML
        """
        try:
            with open(self.__path, "r") as file_:
                config = yaml.load(file_.read(), Loader=yaml.SafeLoader)
                if index not in config:
                    raise KeyError(f"La clave '{index}' no existe en la configuración")
                return config[index]
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error al analizar el archivo YAML: {e}")
        except Exception as e:
            raise Exception(f"Error al acceder a la configuración: {e}")

    def get(self, index: str, default: Any = None) -> Any:
        """Obtiene un elemento de configuración por su clave, con valor predeterminado.
        
        A diferencia de __getitem__, este método no lanza excepciones si la clave
        no existe, sino que devuelve el valor predeterminado.
        
        Args:
            index (str): Clave del elemento de configuración a obtener
            default (Any, optional): Valor predeterminado si la clave no existe
            
        Returns:
            Any: Valor asociado a la clave o valor predeterminado
        """
        try:
            with open(self.__path, "r") as file_:
                return yaml.load(file_.read(), Loader=yaml.SafeLoader).get(index, default)
        except Exception:
            return default


class Stats(object):
    """Clase para gestionar estadísticas en un archivo YAML.
    
    Esta clase permite leer y escribir estadísticas en un archivo YAML,
    proporcionando una interfaz similar a un diccionario.
    """
    
    def __init__(self, filename="stats.yaml"):
        """Inicializa el objeto Stats.
        
        Args:
            filename (str, optional): Ruta al archivo de estadísticas.
                                     Por defecto es "stats.yaml".
        """
        self.file = filename
        self.__data = {}
        
        if os.path.exists(filename):
            try:
                with open(filename, "r") as file_:
                    loaded_data = yaml.load(file_.read(), Loader=yaml.SafeLoader)
                    # Asegurar que data siempre sea un diccionario
                    if loaded_data is not None and isinstance(loaded_data, dict):
                        self.__data = loaded_data
            except Exception as e:
                print(f"Advertencia: No se pudo cargar el archivo de estadísticas: {e}")
                # Continuar con un diccionario vacío

    def get(self, index: str, default: Any = None) -> Any:
        """Obtiene un elemento de estadísticas por su clave, con valor predeterminado.
        
        Args:
            index (str): Clave del elemento de estadísticas a obtener
            default (Any, optional): Valor predeterminado si la clave no existe
            
        Returns:
            Any: Valor asociado a la clave o valor predeterminado
        """
        return self.__data.get(index, default)  # Corregido: faltaba return

    def __getitem__(self, index: str) -> Any:
        """Accede a un elemento de estadísticas por su clave.
        
        Args:
            index (str): Clave del elemento de estadísticas a obtener
            
        Returns:
            Any: Valor asociado a la clave
            
        Raises:
            KeyError: Si la clave no existe en las estadísticas
        """
        if index not in self.__data:
            raise KeyError(f"La clave '{index}' no existe en las estadísticas")
        return self.__data[index]

    def __setitem__(self, index: str, data: Any) -> None:
        """Establece un elemento de estadísticas.
        
        Args:
            index (str): Clave del elemento de estadísticas a establecer
            data (Any): Valor a asociar con la clave
        """
        self.__data[index] = data
        try:
            # Crea el directorio si no existe
            directory = os.path.dirname(self.file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            with open(self.file, "w") as file_:
                yaml.dump(self.__data, file_, default_flow_style=False)
        except Exception as e:
            print(f"Advertencia: No se pudo guardar las estadísticas: {e}")
            # Continuar sin error fatal para no interrumpir el entrenamiento