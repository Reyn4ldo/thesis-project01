"""
Model evaluation module.
Provides comprehensive evaluation metrics and visualizations.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate ML model performance."""
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.results = {}
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained sklearn-compatible model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions if available
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        
        # Add ROC-AUC and PR-AUC if probabilities available
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            metrics["pr_auc"] = average_precision_score(y_test, y_proba)
        except Exception as e:
            logger.warning(f"Could not calculate AUC metrics: {e}")
        
        # Store results
        self.results[model_name] = {
            "metrics": metrics,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
        
        # Log metrics
        logger.info(f"✓ {model_name} evaluation results:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def get_confusion_matrix(
        self, model_name: str
    ) -> np.ndarray:
        """
        Get confusion matrix for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Confusion matrix
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        y_test = self.results[model_name]["y_test"]
        y_pred = self.results[model_name]["y_pred"]
        
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"\nConfusion Matrix for {model_name}:")
        logger.info(f"\n{cm}")
        
        return cm
    
    def get_classification_report(
        self, model_name: str
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Classification report string
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        y_test = self.results[model_name]["y_test"]
        y_pred = self.results[model_name]["y_pred"]
        
        report = classification_report(
            y_test, y_pred,
            target_names=["Non-MDR", "MDR"],
            zero_division=0
        )
        
        logger.info(f"\nClassification Report for {model_name}:")
        logger.info(f"\n{report}")
        
        return report
    
    def calculate_metrics_by_threshold(
        self,
        model_name: str,
        thresholds: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Calculate metrics at different classification thresholds.
        
        Args:
            model_name: Name of the model
            thresholds: Array of thresholds to evaluate (default: 0.1 to 0.9)
            
        Returns:
            DataFrame with metrics at each threshold
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        y_test = self.results[model_name]["y_test"]
        y_proba = self.results[model_name]["y_proba"]
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            results.append({
                "threshold": threshold,
                "accuracy": accuracy_score(y_test, y_pred_thresh),
                "precision": precision_score(y_test, y_pred_thresh, zero_division=0),
                "recall": recall_score(y_test, y_pred_thresh, zero_division=0),
                "f1": f1_score(y_test, y_pred_thresh, zero_division=0)
            })
        
        df_results = pd.DataFrame(results)
        
        logger.info(f"\nMetrics by threshold for {model_name}:")
        logger.info(f"\n{df_results.to_string(index=False)}")
        
        return df_results
