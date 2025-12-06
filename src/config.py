"""Configuration management for AMR surveillance system."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration manager."""
    
    # Project paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    INTERIM_DATA_DIR = DATA_DIR / "interim"
    
    MODELS_DIR = ROOT_DIR / "models"
    ARTIFACTS_DIR = ROOT_DIR / "artifacts"
    DELIVERABLES_DIR = ROOT_DIR / "deliverables"
    LOGS_DIR = ROOT_DIR / "logs"
    
    # Data processing
    MISSING_THRESHOLD = 0.20
    TREAT_INTERMEDIATE_AS_RESISTANT = False
    IMPUTE_METHOD = "mode"
    
    # MDR definition
    MDR_RESISTANT_CLASSES_THRESHOLD = 3
    
    # Model training
    TEST_SIZE = 0.20
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Model algorithms
    ALGORITHMS = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "lightgbm",
        "svm"
    ]
    
    # Evaluation metrics
    PRIMARY_METRIC = "roc_auc"
    METRICS = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    ]
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_WORKERS = int(os.getenv("API_WORKERS", "4"))
    
    # Dashboard configuration
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def load_from_yaml(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all necessary directories exist."""
        for dir_path in [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.INTERIM_DATA_DIR,
            cls.MODELS_DIR,
            cls.ARTIFACTS_DIR,
            cls.DELIVERABLES_DIR,
            cls.LOGS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
