import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pickle
from urllib.parse import urlparse
from sklearn.metrics import (mean_absolute_error,
                             r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from src.utils.logger import logging
from src.utils.exception_handler import Custom_Exception
from src.utils.other_utils import load_object


class ModelEvaluation_MLFlow:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.trained_model_file_path = os.path.join('artifacts', 'model.pkl')
        logging.info("Model Evaluation Begins")

    def eval_metrics(self,y_test,y_pred,is_classification):
        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            try:
                auc_roc = roc_auc_score(y_test, y_pred)  # as it requires probabilities for AUC
            except ValueError:
                auc_roc = None  # Handle case where AUC cannot be computed
            logging.info("Classification Metrics captured")
            return accuracy, precision, recall, f1, auc_roc
        else:
            mse = np.square(y_test - y_pred).mean()
            rmse = np.sqrt(np.square(y_test - y_pred).mean())
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logging.info("Regression Metrics captured")
            return mse, rmse, mae, r2

    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test, y_test = test_array.iloc[:,:-1],test_array.iloc[:,-1]

            preprocessor=load_object(self.preprocessor_obj_file_path)
            model=load_object(self.trained_model_file_path)
            logging.info(" Best Model has been loaded successfullyfrom the Artifacts")

            #mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme  # If we dont set a set_tracking_uri, then it returns a file here otherwise http
            print("tracking_url_type_store :", tracking_url_type_store)
            logging.info("Initiate MLFlow for Experiment tracking")

            with mlflow.start_run():
                y_pred = model.predict(X_test)
                if is_classification:
                    accuracy, precision, recall, f1, auc_roc = self.eval_metrics(y_test, y_pred, is_classification)
                    mlflow.log_metric("Accuracy", accuracy)
                    mlflow.log_metric("Precision", precision)
                    mlflow.log_metric("Recall", recall)
                    mlflow.log_metric("F1_score", f1)
                    if auc_roc is not None:
                        mlflow.log_metric("Auc_Roc", auc_roc)
                    logging.info("Metrics logged for Classification")
                    mlflow.set_tag("Training Classification", "Running the best model")
                else:
                    mse,rmse,mae,r2 =self.eval_metrics(y_test,y_pred,is_classification)
                    mlflow.log_metric("Mse", mse)
                    mlflow.log_metric("Rmse", rmse)
                    mlflow.log_metric("R2", r2)
                    mlflow.log_metric("Mae", mae)
                    logging.info("Metrics logged for Regression")
                    mlflow.set_tag("Training Regression", "Running the best model")
                 # Model registry does not work with file store
                if tracking_url_type_store != "file": # It means we have set_tracking_uri
                    model_info = mlflow.sklearn.log_model(model, "model" 
                                                         # ,registered_model_name="ml_model" # Register once you validate it from UI against paramters for comparison
                                                          )
                    print("Model Info (Not a file) :", model_info.model_uri)
                else:
                    model_info = mlflow.sklearn.log_model(model, "model"
                                                          #,signature= sign
                                                          )
                    print("Model Info (Is a file) when we dont set uri:", model_info.model_uri) 
        except Exception as e:
            logging.error("Error in prediction: %s", str(e))
            raise Custom_Exception(e, sys)
        
# is_classification = True
# train_array = pd.read_csv("artifacts\\train.csv")
# test_array = pd.read_csv("artifacts\\test.csv")

# if __name__ == "__main__":
#     model_evaluation = ModelEvaluation_MLFlow()
#     model_evaluation.initiate_model_evaluation(train_array, test_array)
#     logging.info("Model Evaluation/ MLFlow Experiment tracking  module ran successfully.")
#     print("Model Evaluation/ MLFlow Experiment tracking  module ran successfully.")