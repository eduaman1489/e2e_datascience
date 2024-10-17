from flask import Flask, request,render_template, json, jsonify
import joblib
import pickle
import pandas as pd
import sys
from src.utils.logger import logging
from src.utils.exception_handler import Custom_Exception

app = Flask(__name__)
model = pickle.load(open('artifacts\model.pkl','rb'))

@app.route('/make_api_call',methods=['POST'])
def predict_api():
    try:
        data=request.json['data']
        # new_data = np.array(data)#.reshape(1, -1)
        # print('reshaped data :', new_data)
        output = model.predict(data)
        logging.info("Model output is: %s", str(output[0]))
        return jsonify({"prediction": output.tolist()})  # Convert to list for JSON serialization
    except Exception as e:
        logging.error("Error in prediction: %s", str(e))
        raise Custom_Exception(e, sys)
        return jsonify({"error": str(e)}), 400  # Return error message
    
if __name__ == '__main__':
     app.run(debug=True, port=5002)  # http://127.0.0.1:5002/make_api_call