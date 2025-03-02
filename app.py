from flask import Flask, render_template, jsonify, request
import requests
import pickle  # To load trained ML model
import pandas as pd
from entity.config import TrainingPipelineConfig,ModelTrainerConfig,DataTransformationConfig,MicroTrendAnalysisConfig
from constant.training_pipeline import TARGET_COLUMN,EXCLUDE_COLUMNS
import joblib
import os
from logger import logging
from utils.main_utils import load_object
from components.model_training import ModelTrainer
from components.micro_trend_analysis import MicroTrendAnalysis
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from components.data_transformation import DataTransformation
from sklearn.impute import SimpleImputer
import numpy as np

# Load trained model
app = Flask(__name__, static_folder='static', static_url_path='/static')



# Add this route to serve the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Load models from artifacts/models directory
models_dir = os.path.join('artifacts', 'models')

# Load the model and configuration

# Initialize the training pipeline config first
training_pipeline_config = TrainingPipelineConfig()
model_trainer_config = ModelTrainerConfig(training_pipeline_config)  # Initialize your config
data_transformation_config = DataTransformationConfig(training_pipeline_config)
micro_trend_analysis_config =MicroTrendAnalysisConfig(training_pipeline_config)
iso_forest = joblib.load(os.path.join(models_dir, 'iso_forest.pkl'))
log_reg = joblib.load(os.path.join(models_dir, 'log_reg.pkl'))
preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
imputer= joblib.load(os.path.join(models_dir, 'imputer.pkl'))
logging.info("Successfully loaded models from artifacts/models directory")  # Load the preprocessor

model_training_instance = ModelTrainer(model_trainer_config, None)  # Replace with your actual class initialization
data_transformation_instance=DataTransformation(None,data_transformation_config)
micro_trend_analysis_instance = MicroTrendAnalysis(None,micro_trend_analysis_config)  # Initialize your MicroTrendAnalysis class

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj
try:
    optimal_threshold= joblib.load(os.path.join(os.path.join(models_dir,'optimal_threshold.pkl')))
except :
    optimal_threshold = 0.21 


@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request!", flush=True)
    try:
        # Check if the request contains JSON data
        if request.is_json:
            data = request.get_json()  # Get JSON data from the request
            df = pd.DataFrame(data)  # Convert JSON data to a DataFrame
            logging.info("JSON data received and converted to DataFrame")

        # Check if a file is uploaded
        elif 'file' in request.files:
            file = request.files['file']  # Get the uploaded file

            # Check if the file is a CSV
            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'File must be a CSV'}), 400

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)
            logging.info("CSV file uploaded and read into DataFrame")

        else:
            # If neither JSON nor a file is provided, return an error
            return jsonify({'error': 'No file uploaded or JSON data provided'}), 400
        
        # First, drop EtCO2 column if it exists (since you dropped it during training)
        if 'EtCO2' in df.columns:
            df = df.drop(columns=['EtCO2'])
            logging.info("Dropped EtCO2 column as it was not present during training")
        
        #Add organ disfucntion features
        df= micro_trend_analysis_instance.organ_disfunction(df)
        logging.info("Organ dysfunction features added")

        #Store organ disfucntion features
        organ_dysfunction_results={}
        for patient_id in df['Patient_ID'] .unique():
            patient_data= df[df['Patient_ID']==patient_id]
            organ_dysfunction_results[patient_id] = {
                'Kidney_Dysfunction': patient_data['Kidney_Dysfunction'].any(),
                'Liver_Dysfunction': patient_data['Liver_Dysfunction'].any(),
                'Inflammation': patient_data['Inflammation'].any()
            }

        #Compute micro_trend_features
        sepsis_vitals = ['HR', 'O2Sat', 'MAP', 'Lactate', 'pH', 'Resp', 'Creatinine', 'Bilirubin_total', 'Bilirubin_direct', 'Platelets', 'WBC']
        df=micro_trend_analysis_instance.compute_micro_trends(df,vitals=sepsis_vitals)
        logging.info("Micro trend features computed")

        # Save excluded columns before preprocessing
        excluded_data = df[EXCLUDE_COLUMNS].copy()

        
        # Prepare data for preprocessing (drop excluded columns and target if present)
        input_features= df.drop(columns=EXCLUDE_COLUMNS,errors='ignore')
        if TARGET_COLUMN in input_features.columns:
            input_features = input_features.drop(columns=[TARGET_COLUMN])
        
        # Get the exact feature names from the preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            expected_features = preprocessor.get_feature_names_out()
        else:# Fallback for older scikit-learn versions
            expected_features = preprocessor.get_feature_names() if hasattr(preprocessor, 'get_feature_names') else input_features.columns
        
         # Convert both to lists and ensure they're strings
        expected_features = [str(f).strip() for f in expected_features]
        current_features = [str(f).strip() for f in input_features.columns]
        
        print(f"Preprocessor expects these features: {expected_features}", flush=True)
        print(f"Current features: {input_features.columns.tolist()}", flush=True)

        # Check if they match exactly
        features_match = expected_features == current_features
        print(f"Features match exactly: {features_match}", flush=True)

        if not features_match:
         # Find differences
         print("Features that don't match:", flush=True)
         for i, (exp, cur) in enumerate(zip(expected_features[:min(len(expected_features), len(current_features))])):
          if exp != cur:
            print(f"Position {i}: expected '{exp}', got '{cur}'", flush=True)
    
    # Check for length differences
         if len(expected_features) != len(current_features):
          print(f"Length mismatch: expected {len(expected_features)}, got {len(current_features)}", flush=True)
        
        # Show extra or missing features
          extra = set(current_features) - set(expected_features)
          missing = set(expected_features) - set(current_features)
          if extra:
            print(f"Extra features: {extra}", flush=True)
          if missing:
            print(f"Missing features: {missing}", flush=True)

# Force the exact same feature names
        input_features = input_features.reindex(columns=expected_features, fill_value=0)

# Ensure columns are in exact order
        input_features = input_features[expected_features]
        print(f"Final features before transform: {input_features.columns.tolist()}", flush=True)

         # Now transform the data
        transformed_features = preprocessor.transform(input_features)
        transformed_df = pd.DataFrame(transformed_features, columns=expected_features)
        print(f"Final features after transform: {transformed_df.columns.tolist()}")


        # Add excluded columns back to the transformed DataFrame
        for col in EXCLUDE_COLUMNS:
            print(f"Adding back column: {col}", flush=True)
            transformed_df[col] = excluded_data[col].values
        # Print the columns to see their current order

        print(f"Columns after adding excluded columns: {transformed_df.columns.tolist()}", flush=True)
       
# Handle missing values using the same imputer as during training
        feature_cols = [col for col in transformed_df.columns if col != TARGET_COLUMN]
 
        if transformed_df[feature_cols].isnull().any().any():
         logging.info("Handling missing values in prediction data")
    # Convert to numpy array, apply imputer, then convert back to DataFrame
        imputed_values = imputer.transform(transformed_df[feature_cols])
        transformed_df[feature_cols] = pd.DataFrame(imputed_values, columns=feature_cols, index=transformed_df.index)
        # Predict anomaly scores
        anomaly_scores = iso_forest.predict(transformed_df[feature_cols])
        transformed_df['Anomaly_Score'] = anomaly_scores
        transformed_df['Anomaly_Score'] = transformed_df['Anomaly_Score'].apply(lambda x: 1 if x == -1 else 0)

        # Predict SLS probabilities
        transformed_df['SLS'] = log_reg.predict_proba(transformed_df[feature_cols])[:, 1]
        logging.info("Predictions completed")

        # Check for alerts (e.g., SLS exceeds threshold)
        alerts = []
        for index, row in transformed_df.iterrows():
            if row['SLS'] > optimal_threshold:  # Threshold for alerts
                alerts_message= f"SLS exceeds threshold of {optimal_threshold:.2f}"

                #Add organ dysfucnion information to the alert if available
                patient_id= row['Patient_ID']
                # Convert NumPy types to standard Python types
                if isinstance(patient_id, (np.integer, np.int64)):
                   patient_id = int(patient_id)  # Convert to standard Python int
                elif isinstance(patient_id, (np.floating, np.float64)):
                   patient_id = float(patient_id)  # Convert to standard Python float
                elif isinstance(patient_id, np.ndarray):
                   patient_id = patient_id.item()  # Convert single-element array to scalar

                if patient_id in organ_dysfunction_results:
                   dysfunctions = []
                   if organ_dysfunction_results[patient_id]['Kidney_Dysfunction']:
                        dysfunctions.append('Kidney dysfunction detected')
                   if organ_dysfunction_results[patient_id]['Liver_Dysfunction']:
                        dysfunctions.append('Liver dysfunction detected')
                   if organ_dysfunction_results[patient_id]['Inflammation']:
                        dysfunctions.append('Inflammation detected')
                    
                   if dysfunctions:
                        alerts_message += f". {', '.join(dysfunctions)}"
                
                alerts.append({
                    'Patient_ID': patient_id,
                    'alert': alerts_message,
                    'SLS': float(row['SLS'])
                })
        
        logging.info(f"Alerts generated: {alerts}")


         #Create plot directory if it doesn't exists
        plot_dir = os.path.join('static', 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        #plot_micro_trends_file_path = os.path.join(plot_dir, 'micro_trend_plot_.png')
        plot_sls_file_path = os.path.join(plot_dir, 'sls_evolution_plot.png')  # Define your plot file path

# Plot micro trends for a sample patient
        try:
            patient_id = transformed_df['Patient_ID'].iloc[0]

    # Make sure patient_id is a scalar value, not a list or array
             # Convert NumPy types to standard Python types
            if isinstance(patient_id, (np.integer, np.int64)):
               patient_id = int(patient_id)  # Convert to standard Python int
            elif isinstance(patient_id, (np.floating, np.float64)):
               patient_id = float(patient_id)  # Convert to standard Python float
            elif isinstance(patient_id, np.ndarray):
               patient_id = patient_id.item()  # Convert single-element array to scalar
    
    
    # Plot micro trends
            for vital in sepsis_vitals:
             try:
               plt.figure(figsize=(12, 6))
               print("Creating plot for micro trends plot", flush=True)
               micro_trend_analysis_instance.plot_micro_trends(transformed_df, patient_id,vital=vital)
               print("Called plot_micro_trends function", flush=True)
               plot_micro_trends_file_path = os.path.join(plot_dir,f"micro_trend_plot_{vital}.png")
               plt.savefig(plot_micro_trends_file_path)
               print(f"Saved micro trends plot to {plot_micro_trends_file_path}", flush=True)

               plt.close()
        
             except Exception as e:
               print(f"Error plotting micro trends: {str(e)}", flush=True)
        # Create a simple error plot
               plt.figure(figsize=(12, 6))
               plt.text(0.5, 0.5, f"Error plotting micro trends: {str(e)}", 
               horizontalalignment='center', verticalalignment='center', fontsize=14)
               plt.savefig(plot_micro_trends_file_path)
               plt.close()
            
    
    # Plot SLS evolution
            try:
               plt.figure(figsize=(12, 6))
               print("Created figure for SLS evolution plot", flush=True)
               model_training_instance.plot_sls_evolution(transformed_df, patient_id, optimal_threshold)
               print("Called plot_sls_evolution function", flush=True)
               plt.savefig(plot_sls_file_path)
               print(f"Saved SLS evolution plot to {plot_sls_file_path}", flush=True)
               plt.close()
            except Exception as e:
               print(f"Error plotting SLS evolution: {str(e)}", flush=True)
        # Create a simple error plot
               plt.figure(figsize=(12, 6))
               plt.text(0.5, 0.5, f"Error plotting SLS evolution: {str(e)}", 
               horizontalalignment='center', verticalalignment='center', fontsize=14)
               plt.savefig(plot_sls_file_path)
               plt.close()
        
        except Exception as e:
           print(f"Error in plotting section: {str(e)}", flush=True)
           import traceback
           traceback.print_exc()
    
    # Create simple placeholder plots
           plt.figure(figsize=(12, 6))
           plt.text(0.5, 0.5, f"Error generating plots: {str(e)}", 
           horizontalalignment='center', verticalalignment='center', fontsize=14)
           plt.savefig(plot_micro_trends_file_path)
           plt.close()
    
           plt.figure(figsize=(12, 6))
           plt.text(0.5, 0.5, f"Error generating plots: {str(e)}",horizontalalignment='center', verticalalignment='center', fontsize=14)
           plt.savefig(plot_sls_file_path)
           plt.close()


        # Return the predictions along with alerts and plot file paths

        # Convert organ_dysfunction_results to use string keys
        predictions_list = []
        for record in transformed_df[['Patient_ID', 'SLS', 'Anomaly_Score']].to_dict(orient='records'):
    # Convert each value to a standard Python type
           clean_record = {}
           for key, value in record.items():
              clean_record[key] = convert_to_serializable(value)
           predictions_list.append(clean_record)

# Convert alerts to ensure all values are serializable
        clean_alerts = []
        for alert in alerts:
           clean_alert = {}
           for key, value in alert.items():
              clean_alert[key] = convert_to_serializable(value)
              clean_alerts.append(clean_alert)

# Convert organ_dysfunction_results to ensure all values are serializable
        clean_organ_results = {}
        for patient_id, results in organ_dysfunction_results.items():
           str_patient_id = str(patient_id)
           clean_organ_results[str_patient_id] = {}
           for key, value in results.items():
              clean_organ_results[str_patient_id][key] = convert_to_serializable(value)
        
        vital_plot_urls = {}
        for vital in sepsis_vitals:
            vital_plot_urls[vital] = f'/static/plots/micro_trend_plot_{vital}.png'

        print("Returning JSON response:", {
    'vital_plots': vital_plot_urls,
    'sls_evolution_plot': '/static/plots/sls_evolution_plot.png'
})

# Return the JSON response with all values properly converted
        return jsonify({'predictions': predictions_list,'alerts': clean_alerts,'organ_dysfunction': clean_organ_results,
    'sls_evolution_plot': '/static/plots/sls_evolution_plot.png',  # SLS evolution plot
    'vital_plots': vital_plot_urls})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
