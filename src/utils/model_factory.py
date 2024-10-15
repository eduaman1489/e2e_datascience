from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from src.utils.logger import logging
from src.utils.exception_handler import Custom_Exception
from src.utils.other_utils import save_object

trained_model_file_path = "artifacts/model.pkl"

def evaluate_model(X_train, y_train, X_test, y_test, is_classification=True):
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
    
    save_object(file_path=trained_model_file_path, obj=random_search.best_estimator_)
    logging.info("Best model Object saved to Artifacts")

    return random_search.best_estimator_ if random_search else best_model, model_report
    #return random_search.best_estimator_, model_report





# import pandas as pd
# from sklearn.model_selection import train_test_split
# diabetes = pd.read_csv("artifacts\\raw.csv")
# y = diabetes['Outcome']
# X = diabetes[['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
# X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train = np.load('artifacts/train_array.npy')
# y_train = X_train[:, -1]  # Assuming the last column is the target
# X_train = X_train[:, :-1]  # Features

# X_test = np.load('artifacts/test_array.npy')
# y_test = X_test[:, -1]  # Assuming the last column is the target
# X_test = X_test[:, :-1]  # Features

if __name__ == "__main__":
    best_model, model_report = evaluate_model(X_train, y_train, X_test, y_test, is_classification=True)
    print(best_model)
    print()
    print(model_report)