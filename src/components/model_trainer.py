import os, sys
from dataclasses import dataclass
from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomExceptions
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Split training and test input data')

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models = {
                        "Linear Regression": LinearRegression(),
                        "K-Neighbors Regressor": KNeighborsRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "XGBRegressor": XGBRegressor(), 
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor()
                    }
            
            
            model_report: dict = evaluate_model(X_train=X_train,
                                                y_train=y_train,
                                                X_test=X_test,
                                                y_test=y_test,
                                                models=models,
)
            
            # To get best model score from model_report dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from model_report dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomExceptions("No best model found")
            logging.info(f"Best model is {best_model} with accuracy {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                object=best_model
            )

            predicted = best_model.predict(X_test)

            r2_sqaure = r2_score(y_test,predicted)
            
            return r2_sqaure

        except Exception as e:
            raise CustomExceptions(e,sys)

