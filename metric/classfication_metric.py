from entity.artifact import ClassificationMetricArtifact
from exception import SepsisException
import os,sys
from logger import logging
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score,confusion_matrix,classification_report,precision_score,recall_score

def get_classification_score(y_true,y_pred,zero_division=1) ->ClassificationMetricArtifact:
    try:
        conf_matrix = confusion_matrix(y_true, y_pred)
        model_precision = precision_score(y_true, y_pred,zero_division=zero_division)
        model_recall = recall_score(y_true, y_pred,zero_division=zero_division)
        model_f1_score = f1_score(y_true, y_pred,zero_division=zero_division)
        model_roc_auc_score = roc_auc_score(y_true, y_pred)  # Uses probability scores
        model_average_precision_score=average_precision_score(y_true,y_pred)

        # Generate Classification Report
        model_classification_report = classification_report(y_true, y_pred, zero_division=zero_division)

        # Log the metrics
        logging.info(f"F1-Score: {model_f1_score}")
        logging.info(f"ROC-AUC Score: {model_roc_auc_score}")
        logging.info(f"Average Precision Score: {model_average_precision_score}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}") 
        #lod additional metrics for analysis
        logging.info(f"Precision:\n{model_precision}")
        logging.info(f"Recall:\n{model_recall}")

        #Create ClassificationArtifact wiht accuracy
        classification_metric= ClassificationMetricArtifact(f1_score=model_f1_score,roc_auc_score=model_roc_auc_score,average_precision_score=model_average_precision_score,precision=model_precision,recall=model_recall)
        return classification_metric
    except Exception as e:
        raise SepsisException(e,sys)




