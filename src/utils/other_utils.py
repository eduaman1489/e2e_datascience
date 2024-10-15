from src.utils.logger import logging
from src.utils.exception_handler import Custom_Exception

import sys
import os
import dill

def save_object(file_path, obj:object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Object Pickeled/Dilled ")

    except Exception as e:
        raise Custom_Exception(e, sys)