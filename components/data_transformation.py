from exception import SepsisException
from pandas import DataFrame
from logger import logging
import pandas as pd
import os,sys
from entity.artifact import DataIngestionArtifact,DataTransformationArtifact
from entity.config import DataTransformationConfig
from fancyimpute import IterativeImputer as MICE
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer,SimpleImputer


class DataTransformation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_transformation_config:DataTransformationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation

        """
        try:
            self.data_ingestion_artifact= data_ingestion_artifact
            self.data_transformation_config= data_transformation_config
        
        except Exception as e:
            raise SepsisException(e,sys)
     
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SepsisException(e, sys)
    
    
    def impute_missing_values(self,df:pd.DataFrame)->pd.DataFrame:
        try:

            # Log the initial shape of the DataFrame
            logging.info(f"Initial shape of DataFrame: {df.shape}")
            #identify First Sepsis Onset Per Patient
            vitals = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
            labs = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
                    "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", 
                    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
                    "TroponinI", "Hct","Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
            
            patient_id_col = ["Patient_ID"]

            # 1Split into Pre-Sepsis & Post-Sepsis Phases
            df_pre_sepsis = df[df["SepsisLabel"] == 0].copy()
            df_post_sepsis = df[df["SepsisLabel"] == 1].copy()

            logging.info(f"Pre-Sepsis shape: {df_pre_sepsis.shape}, Post-Sepsis shape: {df_post_sepsis.shape}")

             # Drop rows with >50% missing values
            df_pre_sepsis = df_pre_sepsis.dropna(thresh=len(df_pre_sepsis.columns)//2)
            df_post_sepsis = df_post_sepsis.dropna(thresh=len(df_post_sepsis.columns)//2)
            logging.info("Dropped rows with >50% missing values.")

            # Check for missing columns
            vitals_labs = vitals + labs
            missing_cols_pre = [col for col in vitals_labs if col not in df_pre_sepsis.columns]
            missing_cols_post = [col for col in vitals_labs if col not in df_post_sepsis.columns]

            if missing_cols_pre:
                logging.warning(f"Missing columns in pre-sepsis data: {missing_cols_pre}")
            if missing_cols_post:
                logging.warning(f"Missing columns in post-sepsis data: {missing_cols_post}")

        # Drop missing columns from the list
            vitals_labs = [col for col in vitals_labs if col in df_pre_sepsis.columns and col in df_post_sepsis.columns]
            logging.info(f"Columns used for imputation: {vitals_labs}")
             # Check for columns with all missing values
            all_missing_cols_pre = [col for col in vitals_labs if df_pre_sepsis[col].isnull().all()]
            all_missing_cols_post = [col for col in vitals_labs if df_post_sepsis[col].isnull().all()]

            if all_missing_cols_pre:
                logging.warning(f"Columns with all missing values in pre-sepsis data: {all_missing_cols_pre}")
            if all_missing_cols_post:
                logging.warning(f"Columns with all missing values in post-sepsis data: {all_missing_cols_post}")

                # Drop columns with all missing values from the DataFrame
            df_pre_sepsis = df_pre_sepsis.drop(columns=all_missing_cols_pre)
            df_post_sepsis = df_post_sepsis.drop(columns=all_missing_cols_post)

        # Drop columns with all missing values
            vitals_labs = [col for col in vitals_labs if col not in all_missing_cols_pre and col not in all_missing_cols_post]
            logging.info(f"Columns after dropping all-missing columns: {vitals_labs}")

            # Drop columns that are not in the list
            vitals = [col for col in vitals if col in vitals_labs]
            logging.info(f"Updated vitals list: {vitals}")

            # Pre-Sepsis (Stable Phase) Imputation
            # Forward Fill Vitals
            df_pre_sepsis[vitals] = df_pre_sepsis.groupby(patient_id_col)[vitals].ffill()
            logging.info("Forward filled vitals for pre-sepsis patients.")

            # Apply MICE with PMM for both Pre-Sepsis and Post-Sepsis data
            imputer = MICE(
                max_iter=10,  # Number of iterations
                sample_posterior=True,  # Use PMM
                random_state=42
            )

         # Impute non-sepsis patients
              # Impute Pre-Sepsis data
            df_pre_sepsis_imputed = pd.DataFrame(
                imputer.fit_transform(df_pre_sepsis[vitals_labs]),
                columns=vitals_labs,
                index=df_pre_sepsis.index
            )

            df_pre_sepsis[vitals_labs] = df_pre_sepsis_imputed[vitals_labs]
            logging.info("MICE imputation applied to pre-sepsis data.")

            #foward fill vitals
            df_post_sepsis[vitals]=df_post_sepsis.groupby(patient_id_col)[vitals].ffill()
            logging.info("Forward filled vitals for post-sepsis patients.")

             #If large gaps are found, apply MICE imputation

            # Impute Post-Sepsis data
            df_post_sepsis_imputed = pd.DataFrame(
                imputer.fit_transform(df_post_sepsis[vitals_labs]),
                columns=vitals_labs,
                index=df_post_sepsis.index
            )
            df_post_sepsis[vitals_labs] = df_post_sepsis_imputed[vitals_labs]
            logging.info("MICE imputation applied to post-sepsis data.")


             # Apply KNN Imputation to handle remaining missing values
            knn_imputer = KNNImputer(n_neighbors=4)  # Adjust the number of neighbors as needed

           # Apply KNN Imputation to Pre-Sepsis data
            df_pre_sepsis[vitals_labs] = knn_imputer.fit_transform(df_pre_sepsis[vitals_labs])
            logging.info("KNN imputation applied to pre-sepsis data.")

           # Apply KNN Imputation to Post-Sepsis data
            df_post_sepsis[vitals_labs] = knn_imputer.fit_transform(df_post_sepsis[vitals_labs])
            logging.info("KNN imputation applied to post-sepsis data.")


            #final chcek for msisng values remaining_missing_pre = df_pre_sepsis[vitals_labs].isnull().sum().sum()
            remaining_missing_pre = df_pre_sepsis[vitals_labs].isnull().sum().sum()
            remaining_missing_post = df_post_sepsis[vitals_labs].isnull().sum().sum()

            if remaining_missing_pre > 0 or remaining_missing_post > 0:
                logging.warning(f"Remaining missing values after KNN imputation: Pre-Sepsis={remaining_missing_pre}, Post-Sepsis={remaining_missing_post}")

            """
            # Handle remaining missing values using mean imputation
            mean_imputer = SimpleImputer(strategy='mean')

            # Impute remaining missing values in Pre-Sepsis data
            df_pre_sepsis[vitals_labs] = mean_imputer.fit_transform(df_pre_sepsis[vitals_labs])
            logging.info("Mean imputation applied to pre-sepsis data for remaining missing values.")

            # Impute remaining missing values in Post-Sepsis data
            df_post_sepsis[vitals_labs] = mean_imputer.fit_transform(df_post_sepsis[vitals_labs])
            logging.info("Mean imputation applied to post-sepsis data for remaining missing values.")

            if df_pre_sepsis[vitals_labs].isnull().sum().sum() > 0 or df_post_sepsis[vitals_labs].isnull().sum().sum() > 0:
                logging.error("Missing values still exist after all imputation steps.")
            else:
                logging.info("No missing values remain after imputation.")

        """

        # Combine Pre-Sepsis & Post-Sepsis Data
            df_imputed = pd.concat([df_pre_sepsis, df_post_sepsis]).sort_values(["Patient_ID", "Hour"])
            logging.info(f"Final shape of imputed DataFrame: {df_imputed.shape}")
            print(df_imputed.head())
            return df_imputed

        except Exception as e:
            raise SepsisException(e,sys)
        
    def validate_imputed_data(self, df_original, df_imputed):
        """
        Validate the imputed data to ensure it reflects realistic trends for both sepsis and non-sepsis patients.
        """
        try:
            # 1. Check physiological ranges
            #self.validate_physiological_ranges(df_imputed)

            # 2. Compare distributions
            for col in ["HR", "O2Sat", "Glucose"]:
                self.compare_distributions(df_original, df_imputed, col)

            # 3. Plot temporal trends
            #for patient_id in df_imputed["Patient_ID"].unique()[:3]:  # Check first 3 patients
            #for col in ["HR", "O2Sat"]:
                    #self.plot_temporal_trends(df_imputed, patient_id, col)

            # 4. Check for remaining missing values
            self.check_remaining_missing_values(df_imputed)

        except Exception as e:
            raise SepsisException(e, sys)
    """
    def validate_physiological_ranges(self, df):
        
         Check if imputed values are within physiological ranges.
        
        try:
            # Define physiological ranges
            ranges = {
                "HR": (60, 100),  # Normal heart rate range
                "O2Sat": (95, 100),  # Normal oxygen saturation range
                "Glucose": (70, 110),  # Normal glucose range
            }

            for col, (lower, upper) in ranges.items():
                invalid_values = df[(df[col] < lower) | (df[col] > upper)]
                if not invalid_values.empty:
                    logging.warning(f"Invalid values detected in column {col}: {invalid_values[col].unique()}")

        except Exception as e:
            raise SepsisException(e, sys)
"""
    def compare_distributions(self, df_original, df_imputed, col):
        
        #Compare distributions of observed vs. imputed values.
        
        try:
            plt.figure(figsize=(10, 5))
            sns.histplot(df_original[col], color="blue", label="Original", kde=True)
            sns.histplot(df_imputed[col], color="red", label="Imputed", kde=True)
            plt.title(f"Distribution of {col} (Original vs. Imputed)")
            plt.legend()
            plt.show()

        except Exception as e:
            raise SepsisException(e, sys)
    """
    def plot_temporal_trends(self, df, patient_id, col):
        
        Plot temporal trends for a specific patient and column.
    
        try:

            patient_data = df[df["Patient_ID"] == patient_id]
            plt.figure(figsize=(10, 5))
            plt.plot(patient_data["Hour"], patient_data[col], marker="o", label=col)
            plt.title(f"Temporal Trend for Patient {patient_id} ({col})")
            plt.xlabel("Hour")
            plt.ylabel(col)
            plt.legend()
            plt.show()

        except Exception as e:
            raise SepsisException(e, sys)
     """
    def check_remaining_missing_values(self, df):
        """
        Check for remaining missing values in the DataFrame.
        """
        try:
            missing_values = df.isnull().sum()
            logging.info(f"Remaining missing values:\n{missing_values}")

            rows_with_missing_values = df[df.isnull().any(axis=1)]
            logging.info(f"Rows with missing values:\n{rows_with_missing_values}")

        except Exception as e:
            raise SepsisException(e, sys)

        
    def check_outliers(self,df:pd.DataFrame)->pd.DataFrame:
        try:
            #Capping Outliers Based on Verified Medical Limits
           # Define medical limits for non-sepsis and sepsis patients
           non_sepsis_limits = {
            # Vitals
            "HR": (60, 100),       # Heart Rate (beats per minute)
            "SBP": (90, 120),      # Systolic Blood Pressure (mmHg)
            "DBP": (60, 80),      # Diastolic Blood Pressure (mmHg)
            "Temp": (35, 40),      # Temperature (°C)
            "Resp": (12, 20),      # Respiratory Rate (breaths per minute)
            "O2Sat": (95, 100),    # Oxygen Saturation (%)
            
            # Lab Markers
            "HCO3": (22, 28),      # Bicarbonate (mEq/L)
            "FiO2": (21, 100),     # Fraction of Inspired Oxygen (%)
            "pH": (7.35, 7.45),    # pH
            "PaCO2": (35, 45),     # Partial Pressure of Carbon Dioxide (mmHg)
            "SaO2": (95, 100),     # Oxygen Saturation (%)
            "AST": (10, 40),       # Aspartate Aminotransferase (U/L)
            "BUN": (7, 20),        # Blood Urea Nitrogen (mg/dL)
            "Alkalinephos": (44, 147), # Alkaline Phosphatase (U/L)
            "Calcium": (8.5, 10.2), # Calcium (mg/dL)
            "Chloride": (98, 106),  # Chloride (mEq/L)
            "Creatinine": (0.6, 1.2), # Creatinine (mg/dL)
            "Bilirubin_direct": (0, 0.3), # Direct Bilirubin (mg/dL)
            "Glucose": (70, 110),   # Glucose (mg/dL)
            "Lactate": (0.5, 2.2),  # Lactate (mmol/L)
            "Magnesium": (1.7, 2.2), # Magnesium (mg/dL)
            "Phosphate": (2.5, 4.5), # Phosphate (mg/dL)
            "Potassium": (3.5, 5.0), # Potassium (mEq/L)
            "Bilirubin_total": (0.1, 1.2), # Total Bilirubin (mg/dL)
            "TroponinI": (0, 0.04), # Troponin I (ng/mL)
            "Hct": (34, 50),       # Hematocrit (%)
            "Hgb": (12, 17.2),     # Hemoglobin (g/dL)
            "PTT": (25, 35),        # Partial Thromboplastin Time (seconds)
            "WBC": (4, 11),        # White Blood Cell Count (x10^3/µL)
            "Fibrinogen": (200, 400), # Fibrinogen (mg/dL)
            "Platelets": (150, 450), # Platelets (x10^3/µL)
        }

           sepsis_limits = {
            # Vitals
            "HR": (40, 150),       # Heart Rate can be higher in sepsis
            "SBP": (70, 200),       # Systolic BP can be lower (shock) or higher
            "DBP": (40, 120),       # Diastolic BP can vary widely
            "Temp": (34, 42),       # Temperature can be hypothermic or hyperthermic
            "Resp": (10, 40),        # Respiratory rate can be very high or low
            "O2Sat": (80, 100),     # Oxygen saturation can drop significantly
            
            # Lab Markers
            "HCO3": (15, 30),       # Bicarbonate can be lower in metabolic acidosis
            "FiO2": (21, 100),      # FiO2 can be higher in respiratory failure
            "pH": (7.2, 7.5),       # pH can be lower in acidosis
            "PaCO2": (25, 50),      # PaCO2 can be lower or higher
            "SaO2": (50, 100),      # SaO2 can drop significantly
            "AST": (10, 1000),      # AST can be very high in liver injury
            "BUN": (7, 100),        # BUN can rise in acute kidney injury
            "Alkalinephos": (44, 500), # Alkaline Phosphatase can rise in cholestasis
            "Calcium": (7.0, 11.0), # Calcium can be low in sepsis
            "Chloride": (90, 110),  # Chloride can vary with acid-base disturbances
            "Creatinine": (0.6, 5.0), # Creatinine can rise in acute kidney injury
            "Bilirubin_direct": (0, 5.0), # Direct Bilirubin can rise in liver injury
            "Glucose": (50, 300),   # Glucose can be dysregulated
            "Lactate": (0.5, 20.0), # Lactate can be very high in tissue hypoperfusion
            "Magnesium": (1.0, 3.0), # Magnesium can be low or high
            "Phosphate": (1.0, 6.0), # Phosphate can be low or high
            "Potassium": (3.0, 6.0), # Potassium can be low or high
            "Bilirubin_total": (0.1, 10.0), # Total Bilirubin can rise in liver injury
            "TroponinI": (0, 10.0), # Troponin I can rise in myocardial injury
            "Hct": (20, 60),        # Hematocrit can drop or rise
            "Hgb": (8, 20),         # Hemoglobin can drop or rise
            "PTT": (20, 100),       # PTT can be prolonged in coagulopathy
            "WBC": (2, 50),         # WBC can be very high or low
            "Fibrinogen": (100, 600), # Fibrinogen can drop in DIC
            "Platelets": (50, 500), # Platelets can drop in thrombocytopenia
        }
            
            
         # Split the dataframe into non-sepsis and sepsis patients
           df_non_sepsis = df[df["SepsisLabel"] == 0].copy()
           df_sepsis = df[df["SepsisLabel"] == 1].copy()

           logging.info(f"Non-Sepsis shape: {df_non_sepsis.shape}, Sepsis shape: {df_sepsis.shape}")

          # Cap outliers for sepsis patients
           for col, (lower, upper) in sepsis_limits.items():
            if col in df_sepsis.columns:
                df_sepsis[col] = np.clip(df_sepsis[col], lower, upper)
                logging.info(f"Capped outliers for {col} in sepsis data.")

        # Cap outliers for non-sepsis patients
           for col, (lower, upper) in non_sepsis_limits.items():
            if col in df_non_sepsis.columns:
                df_non_sepsis[col] = np.clip(df_non_sepsis[col], lower, upper)
                logging.info(f"Capped outliers for {col} in non-sepsis data.")

        # Merge the data back together
           df_capped = pd.concat([df_non_sepsis, df_sepsis]).sort_values(["Patient_ID", "Hour"])
           return df_capped

        except Exception as e:
         raise SepsisException(e, sys)
        
    
            
    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            train_df =DataTransformation.read_data(self.data_ingestion_artifact.trained_file_path)
            val_df= DataTransformation.read_data(self.data_ingestion_artifact.val_file_path)
            #test_df=DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)
              # Log original shapes
            logging.info(f"Original train shape: {train_df.shape}")
            logging.info(f"Original validation shape: {val_df.shape}")
            #logging.info(f"Original test shape: {test_df.shape}")

            train_new= self.impute_missing_values(train_df)
            val_new=self.impute_missing_values(val_df)
            #test_new= self.impute_missing_values(test_df)
            
            # Log shapes after imputation
            logging.info(f"Train shape after imputation: {train_new.shape}")
            logging.info(f"Validation shape after imputation: {val_new.shape}")
            #logging.info(f"Test shape after imputation: {test_new.shape}")
            
           #Handle Outliers
            train_final= self.check_outliers(train_new)
            val_final=self.check_outliers(val_new)
            #test_final= self.check_outliers(test_new)


            #Validate imputed data
            self.validate_imputed_data(train_df,train_final)
            self.validate_imputed_data(val_df,val_final)
            #self.validate_imputed_data(test_df,test_final)
            
           

            # Ensure they are DataFrames before saving
            train_transformed = pd.DataFrame(train_final)
            val_transformed = pd.DataFrame(val_final) 
            #test_transformed = pd.DataFrame(test_final)
            
            dir_path_train= os.path.dirname(self.data_transformation_config.transformed_train_file_path)
            dir_path_val= os.path.dirname(self.data_transformation_config.transformed_val_file_path)
            os.makedirs(dir_path_train, exist_ok=True)
            os.makedirs(dir_path_val, exist_ok=True)
            #os.makedirs(self.data_transformation_config.transformed_test_file_path, exist_ok=True)

            # Save the transformed data as CSV
            train_transformed.to_csv(self.data_transformation_config.transformed_train_file_path, index=False,header=True)
            val_transformed.to_csv(self.data_transformation_config.transformed_val_file_path, index=False,header=True)
            #test_transformed.to_csv(self.data_transformation_config.transformed_test_file_path, index=False)
            
           
            #preparing artifact
            data_transformation_artifact =DataTransformationArtifact( 
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_val_file_path=self.data_transformation_config.transformed_val_file_path
                #transformed_test_file_path= self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SepsisException(e,sys)
            
    
        
    





    






