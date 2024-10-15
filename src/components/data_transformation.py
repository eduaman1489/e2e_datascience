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

from dataclasses import dataclass
import os
import dill
    
train_path = "artifacts/train.csv"
test_path = "artifacts/test.csv"

all_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
numerical_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
target_col = "Outcome"

@dataclass
class FeatureEngineering: 
    def __init__(self):
        self.train_data_path = os.path.join("artifacts", "train_array.npy")
        self.test_data_path = os.path.join("artifacts", "test_array.npy")
        self.train_data_path_csv = os.path.join("artifacts", "train_array.csv")
        self.test_data_path_csv = os.path.join("artifacts", "test_array.csv")
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

    def data_transformation(self):
        try:
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformer = ColumnTransformer(transformers=[
                ('numerical_pipeline', numerical_pipeline, numerical_cols)
            ])
            logging.info("Feature Engineering Object Build")
            return transformer
        except Exception as e:
            logging.info("Exception occurred during Feature Engineering")
            raise Custom_Exception(e, sys)
    
    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test dataframe loaded")

            transformer_object = self.data_transformation()
            

            input_feature_train = train_df[numerical_cols]
            target_feature_train = train_df[target_col]

            input_feature_test = test_df[numerical_cols]
            target_feature_test = test_df[target_col]

            input_feature_train_array = transformer_object.fit_transform(input_feature_train)
            input_feature_test_array = transformer_object.transform(input_feature_test)
            logging.info("Fitting & transforming the data using Transformer object")

            train_array = np.c_[input_feature_train_array, np.array(target_feature_train)]
            test_array = np.c_[input_feature_test_array, np.array(target_feature_test)]
            logging.info("Concatenation of train and test array done after transformation")

            np.save(self.train_data_path, train_array)
            np.save(self.test_data_path, test_array)
            logging.info("Train and test datasets saved as Numpy array")

            train_df = pd.DataFrame(train_array,columns=[numerical_cols+[target_col]])
            test_df = pd.DataFrame(test_array,columns=[numerical_cols+[target_col]])

            train_df[target_col] = train_df[target_col].astype(int) # Converting to int , because concatenation make it float
            test_df[target_col] = test_df[target_col].astype(int)

            train_df.to_csv(self.train_data_path_csv, index=False)
            test_df.to_csv(self.test_data_path_csv, index=False)
            logging.info("Train and test datasets saved as CSV")

            save_object(file_path=self.preprocessor_obj_file_path, obj=transformer_object)
            logging.info("Transfomer Object saved to Artifacts")

            #return train_array, test_array
            return train_df, test_df
        except Exception as e:
            logging.error("Error in prediction: %s", str(e))
            raise Custom_Exception(e, sys)

# if __name__ == "__main__":
#     feature_engineering = FeatureEngineering()
#     train_array, test_array = feature_engineering.initialize_data_transformation(train_path, test_path)
#     logging.info("Feature Engineering Module ran successfully.")
#     print("Feature Engineering Module ran successfully.")
