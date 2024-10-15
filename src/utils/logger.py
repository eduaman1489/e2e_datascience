import logging
import os
from datetime import datetime

log_filename=f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
log_path_folder=os.path.join(os.getcwd(),"logs") # Build a new folder called logs inside directory structure
os.makedirs(log_path_folder,exist_ok=True)
complete_log_path=os.path.join(log_path_folder,log_filename) # Put the file inside logs folder
logging.basicConfig(level=logging.INFO, 
                    filename=complete_log_path,
                    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
#logging.info("This is to test this script")