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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

@dataclass
class ModelTraining: 
    def __init__(self):
        self.trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
    def initialize_data_training(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (train_array.iloc[:,:-1],train_array.iloc[:,-1],
                                                test_array.iloc[:,:-1],test_array.iloc[:,-1])
            logging.info("Checking if its a Classification or Regression problem")
            if is_classification:
                models = {
                    'RandomForest': RandomForestClassifier(),
                    'Ada_Boost':AdaBoostClassifier(),
                    'LogisticRegression': LogisticRegression(),
                    'SupportVectorMachine':SVC()
                }
            else:
                models = {
                    'LinearRegression': LinearRegression(),
                    'Lasso': Lasso()
                }

            model_report = {}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Check for regression & classification problem)
                #if hasattr(model, "score"):  # Check if model has a score method , then classification
                if is_classification:  # Check if model has a score method , then classification
                    score = model.score(X_test, y_test)
                    model_report[model_name] = score
                else:
                    mse = np.square(y_test - y_pred).mean() # Else Regression
                    model_report[model_name] = -mse  # We store negative MSE to find the best score

            print("Model Report:", model_report)
            best_model_name = max(model_report, key=model_report.get) if is_classification else min(model_report, key=model_report.get)
            best_model = models[best_model_name]
            print(f"Best Model: {best_model_name} with Score: {model_report[best_model_name]}")
            logging.info(f"Best Model: {best_model_name} with Score: {model_report[best_model_name]}")
            param_distributions = {}
            if is_classification:
                param_distributions = {
                    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
                    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
                    'LogisticRegression': {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]},
                    'SupportVectorMachine': {'C': [0.1, 1, 10, 100, 1000],  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
                }
            else:
                param_distributions = {
                    'LinearRegression': {},
                    'Lasso': {'alpha': np.logspace(-4, 4, 20)}
                }
            
            random_search = None
            if best_model_name in param_distributions:
                random_search = RandomizedSearchCV(best_model, param_distributions[best_model_name],
                                                n_iter=10, cv=2, 
                                                scoring='accuracy' if is_classification else 'neg_mean_squared_error',
                                                random_state=42)
                logging.info("Finding best model using random_search")
                random_search.fit(X_train, y_train)

                if not is_classification:
                    print(f"Best parameters for {best_model_name}: {random_search.best_params_}")
                    print(f"Best score from RandomizedSearchCV (Regression) is: {best_model_name}  {-random_search.best_score_}")
                else:
                    print(f"Best parameters for {best_model_name}: {random_search.best_params_}")
                    print(f"Best score from RandomizedSearchCV (Classification) is: {best_model_name} {random_search.best_score_}")
            logging.info(f"Model Report: {model_report}")
            logging.info(f"Best Model found via RandomSearchCV: {best_model_name} with params: {random_search.best_estimator_}")
            print(f"Best Model found via RandomSearchCV: {best_model_name} with params: {random_search.best_estimator_}")
            
            save_object(file_path=self.trained_model_file_path, obj=random_search.best_estimator_)
            logging.info("Best model Object saved to Artifacts")

            return random_search.best_estimator_ if random_search else best_model, model_report
            #return random_search.best_estimator_, model_report
        
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise Custom_Exception(e, sys)
                
# train_array = pd.read_csv("artifacts\\train.csv")
# test_array = pd.read_csv("artifacts\\test.csv")
# is_classification = True

# if __name__ == "__main__":
#     model_training = ModelTraining()
#     best_model, model_report = model_training.initialize_data_training(train_array, test_array)
#     logging.info("Model Training Module ran successfully & best model saved")
#     print("Model Training  Module ran successfully.")
