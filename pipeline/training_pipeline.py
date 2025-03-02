import os,sys
from logger import logging
from exception import SepsisException
from entity.config import TrainingPipelineConfig,DataIngestionConfig,DataTransformationConfig,ModelTrainerConfig,MicroTrendAnalysisConfig
from components.data_ingestion import DataIngestion
from entity.artifact import DataIngestionArtifact,DataTransformationArtifact,ModelTrainerArtifact,MicroTrendArtifact
from exception import SepsisException
from logger import logging
from components.data_transformation import DataTransformation
from components.micro_trend_analysis import MicroTrendAnalysis
from components.model_training import ModelTrainer



class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        
       

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except  Exception as e:
            raise SepsisException(e,sys)
        
    
        
    def start_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact)->DataTransformationArtifact:
        try:
            data_transformation_config= DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data transformation")
            data_transformation= DataTransformation(data_ingestion_artifact=data_ingestion_artifact,data_transformation_config=data_transformation_config)
            data_transformation_artifact= data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed and artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise SepsisException(e,sys)
    
    def start_micro_trend_analysis(self,data_transformation_artifact:DataTransformationArtifact)->MicroTrendArtifact:
        try:
            micro_trend_analysis_config= MicroTrendAnalysisConfig(training_pipeline_config=self.training_pipeline_config)
            micro_trend_analysis =MicroTrendAnalysis(data_transformation_artifact=data_transformation_artifact,micro_trend_analysis_config=micro_trend_analysis_config)
            micro_trend_artifact= micro_trend_analysis.initiate_micro_trend_analysis()
            logging.info(f"Micro trend analysis completed and artifact: {micro_trend_artifact}")
            return micro_trend_artifact
        except Exception as e:
            raise SepsisException(e,sys)
        
    
    def start_model_training(self,micro_trend_artifact:MicroTrendArtifact)->ModelTrainerArtifact:
        try:
            model_trainer_config= ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer= ModelTrainer(micro_trend_artifact=micro_trend_artifact,model_trainer_config=model_trainer_config)
            model_trainer_artifact= model_trainer.initiate_model_training()
            logging.info(f"Model training completed and artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SepsisException(e,sys)
        
    
    
    def run_pipeline(self):
        try:
            data_ingestion_artifact: DataIngestionArtifact=self.start_data_ingestion()
            data_transformation_artifact:DataTransformationArtifact= self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact)
            micro_trend_artifact:MicroTrendArtifact= self.start_micro_trend_analysis(data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact:ModelTrainerArtifact= self.start_model_training(micro_trend_artifact=micro_trend_artifact)
        except Exception as e:
            raise SepsisException(e,sys)
    