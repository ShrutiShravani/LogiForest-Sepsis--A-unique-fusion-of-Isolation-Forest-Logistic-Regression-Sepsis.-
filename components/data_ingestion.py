from exception import SepsisException
from pandas import DataFrame
from logger import logging
import pandas as pd
import os,sys
from entity.artifact import DataIngestionArtifact
from entity.config import DataIngestionConfig
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.csv_file_path= self.data_ingestion_config.source_file_path
        except Exception as e:
            raise SepsisException(e, sys)
        
    def fetch_patient_data(self, source="csv", csv_file_path=None):
        try:
            logging.info("Exporting data from EHR to feature store")
            
            if source == "csv":
                if not csv_file_path:
                    csv_file_path = self.csv_file_path
                if not csv_file_path:
                    raise SepsisException("CSV path not provided")
                
                #Verify if path exits
                if not os.path.exists(csv_file_path):
                    raise SepsisException("CSV file path does not exist")
                
                logging.info("Reading data from CSV")
                df = pd.read_csv(csv_file_path)
                os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_file_path),exist_ok=True)
                df.to_csv(self.data_ingestion_config.feature_store_file_path, index=False,header=True)
                logging.info("Data saved to artifact location")
                return df
        except Exception as e:
            raise SepsisException(e, sys)
        
    def split_data_as_train_test(self, df:DataFrame) -> None:
        """
        Feature store dataset will be split into train,validation and test file
        """

        try:
            unique_patients= df["Patient_ID"].unique()
            train_patients, test_patients = train_test_split(
                unique_patients, test_size=0.2, random_state=42, stratify=df.groupby("Patient_ID")["SepsisLabel"].max()
            )
          
            train_patients, val_patients = train_test_split(
                train_patients, test_size=0.2, random_state=42, stratify=df[df["Patient_ID"].isin(train_patients)].groupby("Patient_ID")["SepsisLabel"].max()
            )
            
            train_set = df[df["Patient_ID"].isin(train_patients)]
            val_set = df[df["Patient_ID"].isin(val_patients)]
            test_set = df[df["Patient_ID"].isin(test_patients)]
    

            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)


            val_dir_path= os.path.dirname(self.data_ingestion_config.validation_file_path)
            test_dir_path= os.path.dirname(self.data_ingestion_config.testing_file_path)

            os.makedirs(dir_path, exist_ok=True)
            os.makedirs(val_dir_path,exist_ok=True)
            os.makedirs(test_dir_path,exist_ok=True)

            logging.info(f"Exporting train,validation and test file path.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            val_set.to_csv(
                self.data_ingestion_config.validation_file_path,index=False,header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise SepsisException(e,sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion...")
            
            # Fetch data from CSV
            df=self.fetch_patient_data(source="csv", csv_file_path=self.csv_file_path)
            logging.info("Data fetched successfully from CSV")
            self.split_data_as_train_test(df= df)
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,val_file_path=self.data_ingestion_config.validation_file_path,test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info("Data Ingestion completed")
            return data_ingestion_artifact
        except Exception as e:
            raise SepsisException(e, sys)