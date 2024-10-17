import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.utils.logger import logging
from src.utils.exception_handler import Custom_Exception
from src.utils.other_utils import save_object
#from src.utils.model_factory import evaluate_model
from dataclasses import dataclass
import os
import dill

from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.svm import SVC 
from sklearn.metrics import (mean_absolute_error,
                             r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import warnings
warnings.filterwarnings("ignore")

@dataclass
class ModelTraining: 
    def __init__(self,experiment_name):
        self.trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
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

    def initialize_data_training(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (train_array.iloc[:,:-1],train_array.iloc[:,-1],
                                                test_array.iloc[:,:-1],test_array.iloc[:,-1])
            signature=infer_signature(X_train,y_train)
                                           
            models = {
                True: {
                    'RandomForestClassifier': (RandomForestClassifier(), {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30]
                    }),
                    'Ada_Boost': (AdaBoostClassifier(), {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 1.0]
                    }),
                    # 'LogisticRegression': (LogisticRegression(), {
                    #     "C": np.logspace(-3, 3, 7),
                    #     "penalty": ["l1", "l2"]
                    # }),
                    'SupportVectorMachine': (SVC(), {
                        'C': [0.1, 1, 10, 100, 1000],
                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                        'kernel': ['rbf']
                    })
                },
                False: {
                    'LinearRegression': (LinearRegression(), {}),
                    'RandomForestRegressor': (RandomForestRegressor(), {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    })
                }
            }
            
            #mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme  # If we dont set a set_tracking_uri, then it returns a file here otherwise http
            print("tracking_url_type_store :", tracking_url_type_store)
            logging.info("Initiate MLFlow for Experiment tracking")

            model_report = {} 
            for model_name, (model, param_dist) in models[is_classification].items():
                with mlflow.start_run(run_name=model_name):
                    print(f"Currently running {model_name}")
                    random_search = RandomizedSearchCV(
                    model,
                    param_dist,
                    n_iter=5,
                    cv=5,
                    scoring='accuracy' if is_classification else 'neg_mean_squared_error',
                    random_state=42
                )
                    random_search.fit(X_train, y_train)
                    best_model = random_search.best_estimator_
                    best_params = random_search.best_params_
                    print(f"Best hyperparams for {model_name} is :", best_params)
                    #logging.info(f"Best hyperparams for {model_name} is :", {str(best_params)})
                    
                    for param, value in best_params.items():
                        mlflow.log_param(param, value)
                    logging.info("All hyperparams logged")
                    y_pred = best_model.predict(X_test)

                    if is_classification:
                        accuracy, precision, recall, f1, auc_roc = self.eval_metrics(y_test, y_pred, is_classification)
                        mlflow.log_metric("Accuracy", accuracy)
                        mlflow.log_metric("Precision", precision)
                        mlflow.log_metric("Recall", recall)
                        mlflow.log_metric("F1_score", f1)
                        if auc_roc is not None:
                            mlflow.log_metric("Auc_Roc", auc_roc)
                        logging.info("Metrics logged for Classification")
                        mlflow.set_tag("Training Classification", f"Using {model_name}")
                    else:
                        mse, rmse, mae, r2 = self.eval_metrics(y_test, y_pred, is_classification)
                        mlflow.log_metric("Mse", mse)
                        mlflow.log_metric("Rmse", rmse)
                        mlflow.log_metric("R2", r2)
                        mlflow.log_metric("Mae", mae)
                        logging.info("Metrics logged for Regression")
                        mlflow.set_tag("Training Regression", f"Using {model_name}")
                    
                    if tracking_url_type_store != "file": # It means we have set_tracking_uri
                        model_info = mlflow.sklearn.log_model(model, "model" 
                                                            # ,registered_model_name="ml_model" # Register once you validate it from UI against paramters for comparison
                                                            )
                        print("Model Info (Not a file) :", model_info.model_uri)
                    else:
                        model_info = mlflow.sklearn.log_model(model, "model"
                                                             ,signature= signature
                                                              )
                        print("Model Info (Is a file) when we dont set uri:", model_info.model_uri) 
                    print("*"*80)
        except Exception as e:
            logging.error("Error in prediction: %s", str(e))
            raise Custom_Exception(e, sys)
       
# is_classification = True
# train_array = pd.read_csv("artifacts\\train.csv")
# test_array = pd.read_csv("artifacts\\test.csv")

# if __name__ == "__main__":
#     model_training = ModelTraining(experiment_name="Experiment # 1")
#     model_training.initialize_data_training(train_array, test_array)
#     logging.info("Model Training Module ran successfully & best model saved")
#     print("Model Training  Module ran successfully.")
