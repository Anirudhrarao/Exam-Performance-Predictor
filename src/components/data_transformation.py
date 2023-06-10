import os,sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomExceptions
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
            This method will do data transformation on categorical and numerical feature
        '''
        try:
            numerical_features = ["writing_score","reading_score"]
            categorical_features = [
                                'gender',
                                'race_ethnicity',
                                'parental_level_of_education',
                                'lunch',
                                'test_preparation_course']
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("onehotencoder",OneHotEncoder())
                ]
            )

            logging.info(f"Numerical columns {numerical_features}")
            logging.info(f"Categorical columns {categorical_features}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_features),
                    ("cat_pipeline",categorical_pipeline,categorical_features)
                ]
            )

            return preprocessor
             
        except Exception as e:
            raise CustomExceptions(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Training and Testing csv file readed as dataframe')
            logging.info('Obtaining or saving preprocessor object')

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_features = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f'Saving preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                object = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomExceptions(e,sys)
        