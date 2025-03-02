from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str
    val_file_path:str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    #transformed_test_file_path:str
    transformed_val_file_path:str



@dataclass
class MicroTrendArtifact:
    train_micro_trend_file_path:str
    val_micro_trend_file_path:str
    #test_micro_trend_file_path:str
    transformed_object_file_path:str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    roc_auc_score: float
    average_precision_score: float
    precision: float
    recall:float
@dataclass
class ModelTrainerArtifact:
    train_accuracy: ClassificationMetricArtifact
    val_accuracy: ClassificationMetricArtifact
    iso_forest_model_path:str
    log_reg_model_path:str
