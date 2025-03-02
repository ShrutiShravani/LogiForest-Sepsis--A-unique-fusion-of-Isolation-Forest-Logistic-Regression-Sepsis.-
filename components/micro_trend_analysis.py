from logger import logging
from exception import SepsisException
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from entity.config import MicroTrendAnalysisConfig
from entity.artifact import DataIngestionArtifact,DataTransformationArtifact,MicroTrendArtifact
import os,sys
from constant.training_pipeline import DROP_COLUMN,TARGET_COLUMN,EXCLUDE_COLUMNS
from utils.main_utils import save_object,get_model_directory
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from sklearn.impute import SimpleImputer

class MicroTrendAnalysis:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,micro_trend_analysis_config:MicroTrendAnalysisConfig):
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.micro_trend_analysis_config= micro_trend_analysis_config
        except Exception as e:
            raise SepsisException(e,sys)
        
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            standard_scaler = StandardScaler()
            imputer = SimpleImputer(strategy='mean')
            preprocessor = Pipeline(
                steps=[
                   ("Imputer",imputer),
                    ("StandardScaler", standard_scaler) #keep every feature in same range and handle outlier
                    ]
            )
            return preprocessor
        except Exception as e:
            raise SepsisException(e,sys)

    
    def organ_disfunction(self,df:pd.DataFrame):
        df['Kidney_Dysfunction'] = df['Creatinine'] > 1.2  # High creatinine indicates kidney issues
        df['Liver_Dysfunction']=(df['Bilirubin_total'] > 1.2) | (df['Bilirubin_direct'] > 0.3)  # Elevated bilirubin = liver stress
        df['Inflammation'] = (df['WBC'] > 12) & (df['Platelets'] < 150)
        return df
    
    def compute_micro_trends(self,df, patient_col='Patient_ID',vitals=None, window_size=3, alpha=0.2):
       """
       Computes rate-of-change (Δ) and acceleration (Δ²) features for sepsis prediction.
       Also applies rolling mean and EMA smoothing to stabilize trends.

      Parameters:
    - df (pd.DataFrame): Input data with patient vitals.
    - patient_col (str): Column representing unique patient ID.
    - window_size (int): Window size for rolling mean smoothing.
    - alpha (float): Smoothing factor for EMA.

    Returns:
    - pd.DataFrame: DataFrame with added features.
    """
       try:
        if vitals is None:
           raise ValueError("Vital signs list cannot be None")

        for vital in vitals:
        # Compute Rate-of-Change (Δ)
          df[f'Δ-{vital}'] = df.groupby(patient_col)[vital].diff()

        # Compute Acceleration (Δ²)
          df[f'Δ²-{vital}'] = df.groupby(patient_col)[f'Δ-{vital}'].diff()

        # Apply Rolling Mean Smoothing to ensure sustained trends are captured.

          df[f'Rolling_{vital}'] = df.groupby(patient_col)[vital].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())

        # Apply Exponential Moving Average (EMA) to detect early warning signs of sepsis progression.

          df[f'EMA_{vital}'] = df.groupby(patient_col)[vital].transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
         
        
        return df
          
       except Exception as e:
          raise SepsisException(e,sys)

        
    def plot_micro_trends(self,df,patient_id,vital,window_size=3, alpha=0.2):
       """
        Plots micro-trends for selected vitals of a given patient.

        Parameters:
        - df (pd.DataFrame): Processed DataFrame with computed trends.
        - patient_id (int): ID of the patient whose trends are to be plotted.
        - vital (str): Vital sign to be plotted.
        - window_size (int): Rolling mean window size.
        - alpha (float): Smoothing factor for EMA.
        """
       try:
         # Plot Trends
          df_patient = df[df['Patient_ID'] == patient_id]
          required_columns = [vital, f'Rolling_{vital}', f'EMA_{vital}']
          for col in required_columns:
            if col not in df_patient.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame. Ensure compute_micro_trends was called correctly.")

          plt.figure(figsize=(12, 6))
          plt.plot(df_patient['Hour'], df_patient[vital], marker='o', linestyle='-', label=f'{vital} (Raw)', color='blue')
          plt.plot(df_patient['Hour'], df_patient[f'Rolling_{vital}'], linestyle='--', label=f'Rolling Mean ({window_size})', color='green')
          plt.plot(df_patient['Hour'], df_patient[f'EMA_{vital}'], linestyle='-', label=f'EMA (α={alpha})', color='red')
          plt.axhline(y=df_patient[vital].mean(), color='gray', linestyle='dotted', label='Baseline')

        # Labels & Legends
          plt.xlabel("Time (Hours)")
          plt.ylabel(vital)
          plt.title(f"{vital} Trends with Rate-of-Change, Rolling Mean & EMA")
          plt.legend()
          plt.grid(True)
          plt.show()
          return df

       except Exception as e:
         raise SepsisException(e,sys)
       
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SepsisException(e, sys)

    def initiate_micro_trend_analysis(self)->MicroTrendArtifact:
        
        try:
            train_df_new =MicroTrendAnalysis.read_data(self.data_transformation_artifact.transformed_train_file_path)
            val_df_new= MicroTrendAnalysis.read_data(self.data_transformation_artifact.transformed_val_file_path)
            #test_df_new=MicroTrendAnalysis.read_data(self.data_transformation_artifact.transformed_test_file_path)

        
            train_organ_dysfunction= self.organ_disfunction(train_df_new)
            val_organ_disfunciton= self.organ_disfunction(val_df_new)
            #test_organ_disfunction= self.organ_disfunction(test_df_new)

            #plot micro trends
              #compute mciro trends
            sepsis_vitals = ['HR', 'O2Sat', 'MAP', 'Lactate', 'pH', 'Resp', 'Creatinine', 'Bilirubin_total','Bilirubin_direct', 'Platelets', 'WBC']

            train_vital_trends = self.compute_micro_trends(train_organ_dysfunction, vitals=sepsis_vitals)
            val_vital_trends=self.compute_micro_trends(val_organ_disfunciton,vitals=sepsis_vitals)
            #test_vital_trends=self.compute_micro_trends(test_organ_disfunction,vitals=sepsis_vitals)

            print("Columns after compute_micro_trends:", train_vital_trends.columns)
            print("Columns after compute_micro_trends:", val_vital_trends.columns)
            
            

            #plot mircro_trend sof a patient to see the trends in chnaging vitals

            train_patient_id = np.random.choice(train_vital_trends['Patient_ID'].unique())
            val_patient_id = np.random.choice(val_vital_trends['Patient_ID'].unique())
            #test_patient_id = np.random.choice(test_vital_trends['Patient_ID'].unique())

            for vital in sepsis_vitals:
                self.plot_micro_trends(train_vital_trends,patient_id=train_patient_id,vital=vital)
                
                self.plot_micro_trends(val_vital_trends,patient_id=val_patient_id,vital=vital)
                #self.plot_micro_trends(test_vital_trends,patient_id=test_patient_id,vitals=vitals)


            #Drop unecessary columns
            train_final= train_vital_trends.dropna(subset=DROP_COLUMN)
            val_final= val_vital_trends.dropna(subset=DROP_COLUMN)
            #test_final= test_vital_trends.dropna(subset=DROP_COLUMN)
            
        
            #Standard scaler to ensure all features are in same range
            
            #training dataframe
            input_feature_train_df = train_final.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_final[TARGET_COLUMN]

            #validation dataframe
            input_feature_val_df = val_final.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_val_df = val_final[TARGET_COLUMN]
            
            train_excluded = input_feature_train_df[EXCLUDE_COLUMNS]
            val_excluded = input_feature_val_df[EXCLUDE_COLUMNS]

            # Drop the excluded columns from the input features
            input_feature_train_df = input_feature_train_df.drop(columns=EXCLUDE_COLUMNS)
            input_feature_val_df = input_feature_val_df.drop(columns=EXCLUDE_COLUMNS)
            
            preprocessor=self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_train_df= preprocessor_object.transform(input_feature_train_df)
            transformed_val_df =preprocessor_object.transform(input_feature_val_df)
            #transformed_test_df= preprocessor_object.transform(test_final, exclude_columns=EXCLUDE_COLUMNS)

            # Create a models directory
            models_dir = get_model_directory()
           # Save the preprocessor
            preprocessor_dir = os.path.join(models_dir, 'preprocessor.pkl')
            joblib.dump(preprocessor_object, preprocessor_dir)
  
            logging.info(f"Saved preprocessor to {preprocessor_dir}")
        
            
            # Convert NumPy arrays to Pandas DataFrames
            transformed_train_df = pd.DataFrame(transformed_train_df, columns=preprocessor_object.get_feature_names_out())
            transformed_val_df = pd.DataFrame(transformed_val_df, columns=preprocessor_object.get_feature_names_out())

            # Add the excluded columns back to the transformed DataFrames
            transformed_train_df[EXCLUDE_COLUMNS] = train_excluded.values
            transformed_val_df[EXCLUDE_COLUMNS] = val_excluded.values

            transformed_train_df[TARGET_COLUMN] = target_feature_train_df.values
            transformed_val_df[TARGET_COLUMN] = target_feature_val_df.values

            save_object(self.micro_trend_analysis_config.transformed_object_file_path,preprocessor_object)

            trend_dir_path_train= os.path.dirname(self.micro_trend_analysis_config.micro_trend_train_file_path)
            trend_dir_path_val= os.path.dirname(self.micro_trend_analysis_config.micro_trend_val_file_path)
            os.makedirs(trend_dir_path_train, exist_ok=True)
            os.makedirs(trend_dir_path_val, exist_ok=True)
            
            transformed_train_df.to_csv(self.micro_trend_analysis_config.micro_trend_train_file_path, index=False)
            transformed_val_df.to_csv(self.micro_trend_analysis_config.micro_trend_val_file_path, index=False)
            #transformed_test_df.to_csv(self.micro_trend_analysis_config.micro_trend_test_file_path, index=False)



            #save numpy array data
            micro_trend_artifact=MicroTrendArtifact(
            train_micro_trend_file_path=self.micro_trend_analysis_config.micro_trend_train_file_path,
            val_micro_trend_file_path=self.micro_trend_analysis_config.micro_trend_val_file_path,
            #test_micro_trend_file_path=self.micro_trend_analysis_config.micro_trend_test_file_path,
            transformed_object_file_path=self.micro_trend_analysis_config.transformed_object_file_path
        )
            
            return micro_trend_artifact


        except Exception as e:
            raise SepsisException(e,sys)
        