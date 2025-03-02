from constant import training_pipeline
from datetime import datetime
import os
from constant.training_pipeline import CSV_FILE_PATH

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
         timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
         self.pipeline_name: str = training_pipeline.PIPELINE_NAME
         self.artifact_dir:str =os.path.join(training_pipeline.ARTIFACT_DIR,timestamp)
         self.timestamp:str= timestamp

class DataIngestionConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
          self.source_file_path= CSV_FILE_PATH
          self.data_ingestion_dir:str=os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME)
          self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,training_pipeline.FILE_NAME)
          self.training_file_path = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_DIR_NAME,training_pipeline.TRAIN_FILE_NAME)
          self.testing_file_path =os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_DIR_NAME,training_pipeline.TEST_FILE_NAME)
          self.validation_file_path=os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_DIR_NAME,training_pipeline.VAL_FILE_NAME)
          


class DataTransformationConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
          self.data_transformation_dir:str= os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
          self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRANSFORMED_TRAIN_FILE_PATH)
          #self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,training_pipeline.TEST_FILE_NAME)
          self.transformed_val_file_path: str = os.path.join(self.data_transformation_dir,  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.TRANSFORMED_VAL_FILE_PATH)
          
class MicroTrendAnalysisConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
          self.micro_trend_dir:str= os.path.join(training_pipeline_config.artifact_dir,training_pipeline.MICRO_TREND_DIR_NAME)
          self.micro_trend_train_file_path:str=os.path.join(self.micro_trend_dir,training_pipeline.MICRO_TREND_TRAIN_FILE_NAME)
          self.micro_trend_val_file_path:str=os.path.join(self.micro_trend_dir,training_pipeline.MICRO_TREND_VAL_FILE_NAME)
          #self.micro_trend_test_file_path:str=os.path.join(self.micro_trend_dir,training_pipeline.MICRO_TREND_TEST_FILE_NAME)
          self.transformed_object_file_path:str=os.path.join(self.micro_trend_dir,training_pipeline.MICRO_TREND_TRANSFORMED_OBJECT_DIR,training_pipeline.PREPROCESSING_OBJECT_FILE_NAME)

         

class ModelTrainerConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
          self.model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME)
          self.iso_forest_model_path:str= os.path.join(training_pipeline_config.artifact_dir,training_pipeline.MODEL_TRAINER_DIR_NAME,training_pipeline.ISO_FOREST_MODEL_NAME)
          self.log_reg_model_path:str= os.path.join(training_pipeline_config.artifact_dir,training_pipeline.MODEL_TRAINER_DIR_NAME,training_pipeline.LOG_REG_MODEL_NAME)
          self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
