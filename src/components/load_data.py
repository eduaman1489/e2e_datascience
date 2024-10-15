import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.logger import logging
from src.utils.exception_handler import Custom_Exception
from dataclasses import dataclass
from pathlib import Path
import os

dataset = "https://raw.githubusercontent.com/plotly/datasets/refs/heads/master/diabetes.csv"

@dataclass
class Data_Ingestion: # Build a artifacts folder with file names like raw.csv, train.csv etc
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv") #artifacts\raw.csv
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data = pd.read_csv(dataset)
            logging.info("Reading the dataframe")

            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            data.to_csv(self.raw_data_path, index=False)
            logging.info("Saved raw dataset in artifact folder")

            train_data, test_data = train_test_split(data, test_size=0.25)
            train_data.to_csv(self.train_data_path, index=False)
            test_data.to_csv(self.test_data_path, index=False)
            logging.info("Train and test datasets saved")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logging.error("Error in prediction: %s", str(e))
            raise Custom_Exception(e,sys)
        
# if __name__ == "__main__":
#     ingestion = Data_Ingestion()
#     ingestion.initiate_data_ingestion()
#     logging.info("Data ingesting pipeline ran successfully")
#     print("Data ingesting pipeline ran successfully")