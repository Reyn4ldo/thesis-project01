"""
Model training module for supervised learning.
Trains 6 algorithms and creates a leaderboard for model selection.
"""

import joblib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")


class ModelTrainer:
    """Train and evaluate multiple ML models for MDR prediction."""
    
    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 5,
        n_jobs: int = -1
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random seed for reproducibility
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 uses all cores)
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
    
    def get_default_models(self) -> Dict[str, Any]:
        """
        Get dictionary of default models with reasonable hyperparameters.
        
        Returns:
            Dictionary mapping model name to sklearn estimator
        """
        models = {
            "logistic_regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight="balanced",
                n_jobs=self.n_jobs
            ),
            "decision_tree": DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced"
            ),
            "random_forest": RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=self.n_jobs
            ),
            "svm": SVC(
                random_state=self.random_state,
                class_weight="balanced",
                probability=True,
                kernel="rbf"
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models["xgboost"] = xgb.XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=self.n_jobs,
                eval_metric="logloss",  # Consistent eval metric
                use_label_encoder=False
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models["lightgbm"] = lgb.LGBMClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                class_weight="balanced",
                n_jobs=self.n_jobs,
                verbose=-1
            )
        
        return models
    
    def train_single_model(
        self,
        model_name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Any:
        """
        Train a single model on training data.
        
        Args:
            model_name: Name of the model
            model: Sklearn-compatible estimator
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        
        try:
            model.fit(X_train, y_train)
            logger.info(f"✓ {model_name} training completed")
            return model
        except Exception as e:
            logger.error(f"✗ Error training {model_name}: {e}")
            raise
    
    def cross_validate_model(
        self,
        model_name: str,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_name: Name of the model
            model: Sklearn-compatible estimator
            X: Features
            y: Labels
            
        Returns:
            Dictionary of mean CV scores
        """
        logger.info(f"Cross-validating {model_name} with {self.cv_folds} folds...")
        
        # Define scoring metrics
        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        }
        
        # Perform stratified K-fold CV
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        try:
            cv_results = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                return_train_score=False
            )
            
            # Calculate mean scores
            mean_scores = {
                metric: np.mean(cv_results[f"test_{metric}"])
                for metric in scoring.keys()
            }
            
            # Log results
            logger.info(f"✓ {model_name} CV results:")
            for metric, score in mean_scores.items():
                logger.info(f"  {metric}: {score:.4f}")
            
            return mean_scores
            
        except Exception as e:
            logger.error(f"✗ Error in CV for {model_name}: {e}")
            return {}
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        custom_models: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train all models and evaluate them.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Optional test features for final evaluation
            y_test: Optional test labels
            custom_models: Optional custom models dict (overrides defaults)
            
        Returns:
            Dictionary of trained models
        """
        logger.info("=" * 60)
        logger.info("Training All Models")
        logger.info("=" * 60)
        
        # Get models
        models = custom_models if custom_models else self.get_default_models()
        
        trained_models = {}
        
        for model_name, model in models.items():
            logger.info(f"\n--- {model_name.upper().replace('_', ' ')} ---")
            
            # Cross-validation on training data
            cv_scores = self.cross_validate_model(model_name, model, X_train, y_train)
            self.results[model_name] = cv_scores
            
            # Train on full training set
            trained_model = self.train_single_model(model_name, model, X_train, y_train)
            trained_models[model_name] = trained_model
            
            # Evaluate on test set if provided
            if X_test is not None and y_test is not None:
                from .evaluate import ModelEvaluator
                evaluator = ModelEvaluator()
                test_metrics = evaluator.evaluate_model(
                    trained_model, X_test, y_test, model_name
                )
                self.results[model_name].update({
                    f"test_{k}": v for k, v in test_metrics.items()
                })
        
        self.models = trained_models
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)
        
        return trained_models
    
    def create_leaderboard(self) -> pd.DataFrame:
        """
        Create a leaderboard ranking models by performance.
        
        Returns:
            DataFrame with model rankings
        """
        if not self.results:
            logger.warning("No results available. Train models first.")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        leaderboard = pd.DataFrame(self.results).T
        
        # Calculate composite score (weighted average)
        # Priority: ROC-AUC (0.4) + Recall (0.3) + Balanced Accuracy (0.3)
        if all(col in leaderboard.columns for col in ["roc_auc", "recall", "balanced_accuracy"]):
            leaderboard["composite_score"] = (
                0.4 * leaderboard["roc_auc"] +
                0.3 * leaderboard["recall"] +
                0.3 * leaderboard["balanced_accuracy"]
            )
            # Sort by composite score
            leaderboard = leaderboard.sort_values("composite_score", ascending=False)
        
        logger.info("\n" + "=" * 60)
        logger.info("MODEL LEADERBOARD")
        logger.info("=" * 60)
        logger.info(f"\n{leaderboard.to_string()}")
        
        return leaderboard
    
    def get_best_model(self, metric: str = "composite_score") -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Tuple of (model_name, model)
        """
        leaderboard = self.create_leaderboard()
        
        if leaderboard.empty:
            raise ValueError("No models trained yet")
        
        best_model_name = leaderboard.index[0]
        best_model = self.models[best_model_name]
        
        logger.info(f"\n✓ Best model: {best_model_name}")
        
        return best_model_name, best_model
    
    def save_models(self, output_dir: str) -> None:
        """
        Save all trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = output_path / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save leaderboard
        leaderboard = self.create_leaderboard()
        leaderboard_path = output_path / "leaderboard.csv"
        leaderboard.to_csv(leaderboard_path)
        logger.info(f"Saved leaderboard to {leaderboard_path}")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model
        """
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
