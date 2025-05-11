import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd

from us_visa.logger.logger import logging
from us_visa.exception.exception import CustomException



def save_object(filepath,obj):
    """
    Saves a Python object to a file using pickle.

    Parameters:
    filepath (str): The path (including filename) where the object should be saved.
    obj (Any): The Python object you want to save.

    This function creates the directory if it doesn't exist,
    and writes the object in binary format using the pickle module.
    """

    try:
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info("Exception occured in save_object function utils.utils")
        raise CustomException(e,sys)
    
def load_object(file_path):
    """
    load object from file using pickle

    parameters:
    file_path (str): The path (including filename) from where the object should be loaded.
    return: The Python object loaded from the file.
    """
    try:
        with open(file_path,'rb') as obj:
            return pickle.load(obj)
    except Exception as e:
        logging.info("Exception occured in load_object function utils.utils")
        raise CustomException(e,sys)
    

def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and return its content as a dictionary.
    file_path: str location of the YAML file
    return: dict content of the YAML file
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes the given content to a YAML file.

    Args:
        file_path (str): Path where the YAML file will be saved.
        content (object): The data to write into the file (usually a dictionary or list).
        replace (bool, optional): If True, deletes the existing file before writing. Default is False.

    Raises:
        CustomException: If an error occurs during file creation or writing.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e,sys)

def evaluat_model(X_train,Y_train,X_test,Y_test,models):
    try:
        dict1={}
        for i,model in models.items():
            model.fit(X_train,Y_train)

            y_pred=model.predict(X_test)
            r2=r2_score(Y_test,y_pred)

            dict1[model]=r2
        return max(dict1.items(),key=lambda x:x[1]) 
    except Exception as e:
        raise CustomException(e,sys)



def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
