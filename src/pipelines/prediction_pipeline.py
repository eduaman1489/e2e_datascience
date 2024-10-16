import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.utils.logger import logging
from src.utils.exception_handler import Custom_Exception
from src.utils.other_utils import load_object

class PredictPipeline:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.trained_model_file_path = os.path.join('artifacts', 'model.pkl')

    def predict_method(self,features):
        try:
            preprocessor=load_object(self.preprocessor_obj_file_path)
            model=load_object(self.trained_model_file_path)
            logging.info("Preprocessor & Model successfully loaded")
            features_df = pd.DataFrame([features],columns=input_columns)
            print(features_df.head())
            logging.info("Dataframe built on input features")
            scaled_features = preprocessor.transform(features_df)
            print(scaled_features)
            prediction = model.predict(scaled_features)
            logging.info("Features scaled & prediction done")
            return prediction
        except Exception as e:
            logging.error("Error in prediction: %s", str(e))
            raise Custom_Exception(e, sys)

input_columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
#features = [6,148,72,35,0,33.6,0.627,50] # output 1
#features = [1,85,66,29,0,26.6,0.351,31] # output 0

# if __name__ == "__main__":
#     predict_pipeline = PredictPipeline()
#     prediction = predict_pipeline.predict_method(features)
#     logging.info("Prediction completed successfully")
#     print("Prediction result :", prediction)



# preprocessor_obj_file_path = pickle.load(open("artifacts\preprocessor.pkl","rb"))
# trained_model_file_path = pickle.load(open("artifacts\model.pkl","rb"))
# input_columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
# new_data = pd.DataFrame([features], columns=input_columns)  
# print('New data :', new_data, new_data.shape)
# scaled_data = preprocessor_obj_file_path.transform(new_data)
# print('Scaled data :', scaled_data)
# output = trained_model_file_path.predict(new_data)
# print(output)