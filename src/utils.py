import os,sys
import numpy as np
import pandas as pd
from src.exception import CustomExceptions
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def save_object(file_path,object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(object,file_obj)

    except Exception as e:
        raise CustomExceptions(e,sys)
    


def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            # para = params[list(models.keys())[i]]

            # grid_search = GridSearchCV(
            #     model,
            #     para,
            #     cv=3
            # )

            # model.set_params(**grid_search.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomExceptions(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomExceptions(e,sys)