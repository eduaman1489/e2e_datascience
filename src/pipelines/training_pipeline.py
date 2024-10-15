import os
import sys
import pandas as pd

from src.components.load_data import Data_Ingestion
from src.components.data_transformation import FeatureEngineering
from src.components.model_training import ModelTraining
from src.utils.logger import logging
from src.utils.exception_handler import Custom_Exception

ingestion=Data_Ingestion()
train_data_path,test_data_path=ingestion.initiate_data_ingestion()

feature_engineering=FeatureEngineering()
train_array, test_array = feature_engineering.initialize_data_transformation(train_data_path, test_data_path)

model_training = ModelTraining()
best_model, model_report = model_training.initialize_data_training(train_array, test_array)