ARTIFACT_DIR: str = "artifact"
PIPELINE_NAME: str = "sepsis"
CSV_FILE_PATH= r"C:\Users\AdmiN\Desktop\final early sepsis\Early_sepsis_risk_prediction\dataset\train_dataset.csv"
TARGET_COLUMN="SepsisLabel"
DROP_COLUMN= ["Unit1","Unit2","HospAdmTime","ICULOS"]
EXCLUDE_COLUMNS=["Patient_ID","Hour","Gender","Age"]
NEW_COLUMN=["Patient_ID","Hour","Gender","Age"]
FILE_NAME: str = "sepsis.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
VAL_FILE_NAME:str="val.csv"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

#Data Ingestion related constants
"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str ="Data_Ingestion"
DATA_INGESTION_FEATURE_STORE_DIR :str ="feature_store"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
TRANSFORMED_TRAIN_FILE_PATH:str='train_transformed.csv'
TRANSFORMED_VAL_FILE_PATH:str='val_transformed.csv'



"""
Micro Trend Related constant
"""
 
MICRO_TREND_DIR_NAME:str="micro_trend"
MICRO_TREND_TRAIN_FILE_NAME:str="train_micro_trend.csv"
MICRO_TREND_VAL_FILE_NAME:str="val_micro_trend.csv"
MICRO_TREND_TEST_FILE_NAME:str="test_micro_trend.csv"
MICRO_TREND_TRANSFORMED_OBJECT_DIR: str = "transformed_object"



"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl" 
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05
ISO_FOREST_MODEL_NAME:str="iso_forest.pkl"
LOG_REG_MODEL_NAME:str="log_reg.pkl"