from exception import SepsisException
from logger import logging
import os,sys
import numpy as np
import dill


def save_numpy_array_data(file_path:str,array:np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logging.info(f"Numpy array saved at {file_path}")
    except Exception as e:
        raise SepsisException(e, sys) 

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise SepsisException(e, sys) from e


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise SepsisException(e, sys) from e

def get_model_directory(model_type=None):
    """Create and return path to models directory"""
    try:
        if model_type:
            models_dir = os.path.join('artifacts', 'models', model_type)
        else:
           models_dir = os.path.join('artifacts', 'models')
    
        os.makedirs(models_dir, exist_ok=True)
        return models_dir
    
    except Exception as e:
        raise SepsisException(e,sys)