import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomExceptions

def save_object(file_path,object):
    '''
        Desc: This function will save our pickle file
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(object,file_obj)

    except Exception as e:
        raise CustomExceptions(e,sys)
    


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    '''
        Desc: This function will evaluate our performance of models and than return dict
              of performance
    '''
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            grid = GridSearchCV(model,para,cv=3)
            grid.fit(X_train,y_train)

            model.set_params(**grid.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomExceptions(e, sys)
    
def load_object(file_path):
    '''
        Desc: This method will load our saved pickle file
    '''
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomExceptions(e,sys)