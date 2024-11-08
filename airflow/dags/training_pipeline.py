from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import numpy as np
import pendulum
import sys

from src.components.load_data import Data_Ingestion
from src.components.data_transformation import FeatureEngineering
from src.components.data_transformation import FeatureEngineering
#from src.components.model_training import ModelTraining
from src.components.mlflow_model_training import ModelTraining
from src.utils.logger import logging
from src.utils.exception_handler import Custom_Exception

def start_data_ingestion():
    try:
        data_ingestion = Data_Ingestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully.")
        return {"train_data_path": train_data_path, "test_data_path": test_data_path}
    except Exception as e:
        logging.error("Error during data ingestion:", e)
        raise Custom_Exception(e, sys)

def start_data_transformation(**kwargs):
    try:
        ti = kwargs['ti']
        paths = ti.xcom_pull(task_ids='data_ingestion') # XCom holds the information/variable and pass it from 1 function to aother
        train_data_path = paths['train_data_path']
        test_data_path = paths['test_data_path']

        data_transformation = FeatureEngineering()
        train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
        logging.info("Data transformation completed successfully.")
        return {
            "train_arr": train_arr.tolist(),  
            "test_arr": test_arr.tolist()  
        }
    except Exception as e:
        logging.error("Error during data transformation:", e)
        raise Custom_Exception(e, sys)

def start_model_training(**kwargs):
    try:
        ti = kwargs['ti']
        arrays = ti.xcom_pull(task_ids='data_transformation')
        train_arr = np.array(arrays['train_arr'])  # Convert list back to numpy array
        test_arr = np.array(arrays['test_arr'])  # Convert list back to numpy array
        model_trainer = ModelTraining()
        model_trainer.initiate_model_training(train_arr, test_arr)
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error("Error during model training:", e)
        raise Custom_Exception(e, sys)

with DAG(
    "MLOps_pipeline", #dag_id
    default_args={
        "owner": "airflow",
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
        "start_date": datetime(2024, 1, 17, tzinfo=pendulum.timezone("UTC")),
        "email_on_failure": False,
        "email_on_retry": False,
    },
    description="Training pipeline MLOps",
    schedule_interval="@weekly",  # Weekly schedule
    catchup=False,
    tags=["machine_learning", "classification", "business_problem"],
) as dag:
    
    # Define task within PythonOperators
    data_ingestion_task = PythonOperator(
        task_id='data_ingestion',
        python_callable=start_data_ingestion,
    )

    data_transformation_task = PythonOperator(
        task_id='data_transformation',
        python_callable=start_data_transformation,
        provide_context=True,
    )

    model_training_task = PythonOperator(
        task_id='model_training',
        python_callable=start_model_training,
        provide_context=True,
    )

    # Directed : Set task order/sequence 
    data_ingestion_task >> data_transformation_task >> model_training_task 