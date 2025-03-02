from exception import SepsisException
from pandas import DataFrame
from logger import logging
import pandas as pd
import os,sys
from entity.artifact import ModelTrainerArtifact,MicroTrendArtifact
from entity.config import ModelTrainerConfig
import numpy as np
from metric.classfication_metric import get_classification_score
import seaborn as sns
from utils.main_utils import save_object,get_model_directory
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from constant.training_pipeline import TARGET_COLUMN
from sklearn.impute import SimpleImputer
import joblib

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,micro_trend_artifact:MicroTrendArtifact):
        try:
            self.model_trainer_config= model_trainer_config
            self.micro_trend_artifact=micro_trend_artifact
            
        except Exception as e:
            raise SepsisException(e,sys)
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SepsisException(e, sys)
    
    def train_sepsis_model(self,df,contamination=0.05):
       
       #Trains an Isolation Forest model to detect sepsis anomalies and a Logistic Regression model to calculate Sepsis Latency Score (SLS).

       # Train Isolation Forest for Anomaly Detection
       try:
           logging.info("Training Isolation Forest model")
           feature_cols = [col for col in df.columns if col != TARGET_COLUMN]

           print(f"Total number of feature columns: {len(feature_cols)}", flush=True)
           print(f"Feature columns: {feature_cols}", flush=True)


           iso_forest = IsolationForest(contamination=contamination, random_state=42)
           df['Anomaly_Score'] = iso_forest.fit_predict(df[feature_cols])

    # Convert anomaly scores: -1 (anomalous) -> 1 (high sepsis risk), 1 (normal) -> 0 (low risk)
           df['Anomaly_Score'] = df['Anomaly_Score'].apply(lambda x: 1 if x == -1 else 0)

    # Train Logistic Regression for Sepsis Latency Score (SLS)
           log_reg = LogisticRegression()
           log_reg.fit(df[feature_cols], df['Anomaly_Score'])

    # Predict Sepsis Latency Score (SLS)
           df['SLS'] = log_reg.predict_proba(df[feature_cols])[:, 1] 
           return df, iso_forest, log_reg
       except Exception as e:
           raise SepsisException(e,sys)
     
    def plot_sls_evolution(self, df: pd.DataFrame, patient_id, threshold):
      """
      Plots real-time Sepsis Latency Score (SLS) evolution for a given patient.
      """
      try:
          logging.info(f"Plotting SLS evolution for patient {patient_id}")
          df_patient = df[df['Patient_ID'] == patient_id]
         
         # Calculate risk levels (low, moderate, high)
          low_risk = np.percentile(df['SLS'], 50)  # Median
          moderate_risk = np.percentile(df['SLS'], 80)  # Top 20% risk 
          if threshold is None:
              threshold = np.percentile(df['SLS'], 95)  # Use top 5% as threshold

          plt.figure(figsize=(12, 6))
          plt.plot(df_patient['Hour'], df_patient['SLS'], marker='o', linestyle='-', label='Sepsis Latency Score (SLS)', color='blue')
    
    # Sepsis Threshold
          plt.axhline(y=threshold, color='red', linestyle='--', label=f'Sepsis Alert Threshold ({threshold})')

    # Color-coded Risk Levels
          for i, sls in enumerate(df_patient['SLS']):
           color = 'green' if sls < low_risk else 'yellow' if sls < moderate_risk else 'red'
           plt.scatter(df_patient['Hour'].iloc[i], sls, color=color, s=100)

    # Labels & Legends
          plt.xlabel("Time (Hours)")
          plt.ylabel("Sepsis Latency Score (SLS)")
          plt.title(f"Real-Time Sepsis Risk Evolution for Patient {patient_id}")
          plt.legend()
          plt.grid(True)
          plt.show()

      except Exception as e:
            logging.error("Error plotting SLS evolution")
            raise SepsisException(e, sys)


    def optimal_threshold(self, df: pd.DataFrame):
        #find optimal threshod
            fpr, tpr, thresholds = roc_curve(df['SepsisLabel'], df['SLS'])
            optimal_idx = np.argmax(tpr - fpr)  # Maximize True Positive Rate - False Positive Rate
            optimal_threshold = thresholds[optimal_idx]
            return optimal_threshold

    def initiate_model_training(self)-> ModelTrainerArtifact:
        try:
            # Load transformed training and testing data
            logging.info("Model training started")
            # Load transformed training and testing data
            train_df = ModelTrainer.read_data(self.micro_trend_artifact.train_micro_trend_file_path)
            val_df= ModelTrainer.read_data(self.micro_trend_artifact.val_micro_trend_file_path)
            #test_array = load_numpy_array_data(self.micro_trend_artifact.test_micro_trend_file_path)

            logging.info(f"Loaded train array shape: {train_df.shape}")
            logging.info(f"Loaded test array shape: {val_df.shape}")
            
           # final check on missing values in training and validation data
            feature_cols = [col for col in train_df.columns if col != TARGET_COLUMN]

            if train_df[feature_cols].isnull().any().any():
             logging.warning("Training data contains missing values. Imputing missing values with the mean.")
            else:
                logging.info("No missing values in training data, but fitting imputer for consistency.")

            imputer = SimpleImputer(strategy='mean')
            imputer_fit = imputer.fit(train_df[feature_cols])
            train_df[feature_cols] = imputer_fit.transform(train_df[feature_cols])

            models_dir = get_model_directory()
            imputer_path = os.path.join(models_dir, 'imputer.pkl')
            joblib.dump(imputer_fit, imputer_path)
            logging.info(f"Saved fitted imputer to {models_dir}")


            # final chcek on missing values in validation data using the same imputer
            if val_df[feature_cols].isnull().any().any():
                logging.warning("Validation data contains missing values. Imputing missing values with the mean.")
                val_df[feature_cols] = imputer.transform(val_df[feature_cols])

            # Train Isolation Forest and Logistic Regression models on training data 
            logging.info("Getting Isolation Forest model")
            train_df,iso_forest, log_reg = self.train_sepsis_model(train_df)
            logging.info("Training started")
   
            logging.info("Making predictions")
            
            train_df['SLS_Predicted'] = (train_df['SLS'] >=0.6).astype(int)
            # Check unique values in TARGET_COLUMN
            logging.info("Unique values in TARGET_COLUMN:", train_df[TARGET_COLUMN].unique())

            # Check unique values in SLS_Predicted
            logging.info("Unique values in SLS_Predicted:", train_df['SLS_Predicted'].unique())
            classification_train_metric= get_classification_score(y_true=train_df[TARGET_COLUMN],y_pred=train_df['SLS_Predicted'])
            logging.info("Model training Completed")
            

             # Step 1: Use the trained Isolation Forest to compute anomaly scores for validation data
            val_df['Anomaly_Score'] = iso_forest.predict(val_df.drop(columns=[TARGET_COLUMN]))
            val_df['Anomaly_Score'] = val_df['Anomaly_Score'].apply(lambda x: 1 if x == -1 else 0)

            # Step 2: Use the trained Logistic Regression to predict SLS for validation data
            val_df['SLS'] = log_reg.predict_proba(val_df.drop(columns=[TARGET_COLUMN,'Anomaly_Score']))[:, 1]
            val_df['SLS_Predicted'] = (val_df['SLS'] >= 0.6).astype(int)
            
           # Check unique values in SLS_Predicted for validation data
            logging.info("Unique values in SLS_Predicted (val):", val_df['SLS_Predicted'].unique())


            #Get validation metrics
          
            classification_val_metric = get_classification_score(y_true=val_df[TARGET_COLUMN], y_pred=val_df['SLS_Predicted'])
            logging.info("Model Validation completed")

            #find optimal threshod
            optimal_threshold= self.optimal_threshold(train_df)
            logging.info(f"Optimal threshold: {optimal_threshold}")

            #plot sl
            self.plot_sls_evolution(df=train_df,patient_id=np.random.choice(train_df['Patient_ID'].unique()),threshold=optimal_threshold)
            self.plot_sls_evolution(df=val_df,patient_id=np.random.choice(val_df['Patient_ID'].unique()),threshold=optimal_threshold)

           
           
             #Overfitting and Underfitting
            """
            diff = abs(classification_train_metric.f1_score-classification_val_metric.f1_score)
            
            if diff>self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good ,try to do more experimentation.")
            """
            

            models_dir = get_model_directory()

             # Save the models
            iso_forest_path = os.path.join(models_dir, 'iso_forest.pkl')
            joblib.dump(iso_forest, iso_forest_path)

            log_reg_path = os.path.join(models_dir, 'log_reg.pkl')
            joblib.dump(log_reg, log_reg_path)
        
            logging.info(f"Saved models to {models_dir}")

            #sAVE TO ORIGINAL PATHS
            save_object(self.model_trainer_config.iso_forest_model_path, iso_forest)
            save_object(self.model_trainer_config.log_reg_model_path, log_reg)

            model_trainer_artifact = ModelTrainerArtifact(iso_forest_model_path=self.model_trainer_config.iso_forest_model_path, 
            log_reg_model_path=self.model_trainer_config.log_reg_model_path,
            train_accuracy=classification_train_metric,
            val_accuracy=classification_val_metric)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise SepsisException(e,sys)
