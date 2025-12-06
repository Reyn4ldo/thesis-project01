"""
Feature engineering for AMR surveillance ML models.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for ML modeling."""
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
    
    def engineer_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Engineer features from cleaned data.
        
        Args:
            df: Cleaned DataFrame
            fit: If True, fit encoders/scalers. If False, use existing.
            
        Returns:
            Tuple of (features DataFrame, feature names list)
        """
        logger.info("Engineering features")
        df_features = df.copy()
        
        # Extract antibiotic resistance features (already encoded)
        antibiotic_cols = [col for col in df_features.columns if col.endswith("_encoded")]
        
        # Encode categorical metadata
        categorical_cols = [
            "bacterial_species",
            "sample_source",
            "administrative_region"
        ]
        
        for col in categorical_cols:
            if col in df_features.columns:
                if fit:
                    le = LabelEncoder()
                    df_features[f"{col}_encoded"] = le.fit_transform(
                        df_features[col].fillna("unknown")
                    )
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories efficiently
                        le = self.label_encoders[col]
                        # Map known categories, use -1 for unknown
                        known_categories = set(le.classes_)
                        df_features[f"{col}_encoded"] = df_features[col].fillna("unknown").map(
                            lambda x: le.transform([x])[0] if x in known_categories else -1
                        )
        
        # Select feature columns
        feature_cols = (
            antibiotic_cols +
            [f"{col}_encoded" for col in categorical_cols if col in df_features.columns] +
            ["num_resistant", "num_tested", "mar_index_calculated"]
        )
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df_features.columns]
        
        if fit:
            self.feature_names = feature_cols
        
        logger.info(f"Engineered {len(feature_cols)} features")
        
        return df_features[feature_cols], feature_cols
    
    def prepare_for_modeling(
        self, 
        df: pd.DataFrame,
        target_col: str = "is_mdr",
        fit: bool = True,
        scale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for modeling.
        
        Args:
            df: Input DataFrame with features and target
            target_col: Name of target column
            fit: If True, fit transformers
            scale: If True, apply standard scaling
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Engineer features
        df_features, feature_names = self.engineer_features(df, fit=fit)
        
        # Extract features and target
        X = df_features[feature_names].values
        y = df[target_col].values if target_col in df.columns else None
        
        # Handle missing values in features
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        if scale:
            if fit:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            else:
                if self.scaler is not None:
                    X = self.scaler.transform(X)
        
        logger.info(f"Prepared features: X shape {X.shape}, y shape {y.shape if y is not None else 'None'}")
        
        return X, y, feature_names
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
