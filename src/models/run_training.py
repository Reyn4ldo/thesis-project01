#!/usr/bin/env python
"""
Command-line interface for training all 6 supervised ML models.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.engineer_features import FeatureEngineer
from src.models.train_models import ModelTrainer
from src.models.evaluate import ModelEvaluator


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(
        description="Train supervised ML models for MDR prediction"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to cleaned CSV data file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models/)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data for test set (default: 0.20)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 60)
        logger.info("AMR Supervised Learning Pipeline")
        logger.info("=" * 60)
        
        # Load cleaned data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Check for target column
        if "is_mdr" not in df.columns:
            raise ValueError("Target column 'is_mdr' not found in data")
        
        # Log class distribution
        mdr_count = df["is_mdr"].sum()
        logger.info(f"Class distribution: {mdr_count} MDR ({mdr_count/len(df):.1%}), "
                   f"{len(df)-mdr_count} non-MDR ({(len(df)-mdr_count)/len(df):.1%})")
        
        # Feature engineering
        logger.info("\nEngineering features...")
        feature_engineer = FeatureEngineer()
        X, y, feature_names = feature_engineer.prepare_for_modeling(
            df, target_col="is_mdr", fit=True, scale=True
        )
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Number of features: {len(feature_names)}")
        
        # Train-test split
        logger.info(f"\nSplitting data (test_size={args.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train models
        logger.info("\n" + "=" * 60)
        logger.info("Training Models")
        logger.info("=" * 60)
        
        trainer = ModelTrainer(
            random_state=args.random_state,
            cv_folds=args.cv_folds,
            n_jobs=-1
        )
        
        trained_models = trainer.train_all_models(
            X_train, y_train,
            X_test, y_test
        )
        
        # Create leaderboard
        leaderboard = trainer.create_leaderboard()
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model()
        
        # Detailed evaluation of best model
        logger.info("\n" + "=" * 60)
        logger.info(f"Detailed Evaluation of Best Model: {best_model_name}")
        logger.info("=" * 60)
        
        evaluator = ModelEvaluator()
        test_metrics = evaluator.evaluate_model(
            best_model, X_test, y_test, best_model_name
        )
        
        # Confusion matrix
        evaluator.get_confusion_matrix(best_model_name)
        
        # Classification report
        evaluator.get_classification_report(best_model_name)
        
        # Save models
        logger.info("\n" + "=" * 60)
        logger.info("Saving Models")
        logger.info("=" * 60)
        
        output_dir = Path(args.output_dir)
        trainer.save_models(output_dir)
        
        # Save feature engineer
        import joblib
        feature_engineer_path = output_dir / "feature_engineer.pkl"
        joblib.dump(feature_engineer, feature_engineer_path)
        logger.info(f"Saved feature engineer to {feature_engineer_path}")
        
        # Save feature names
        feature_names_path = output_dir / "feature_names.txt"
        with open(feature_names_path, "w") as f:
            f.write("\n".join(feature_names))
        logger.info(f"Saved feature names to {feature_names_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ Model training completed successfully")
        logger.info("=" * 60)
        logger.info(f"\nBest model: {best_model_name}")
        logger.info(f"Test ROC-AUC: {test_metrics.get('roc_auc', 'N/A'):.4f}")
        logger.info(f"Test Recall: {test_metrics.get('recall', 'N/A'):.4f}")
        logger.info(f"Test F1: {test_metrics.get('f1', 'N/A'):.4f}")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
