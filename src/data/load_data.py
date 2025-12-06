"""
Data loader module for AMR surveillance data.
Handles loading and initial validation of raw CSV data.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate AMR surveillance data from CSV files."""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the raw CSV data file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    def load(self) -> Tuple[pd.DataFrame, str]:
        """
        Load raw data from CSV and compute hash for provenance.
        
        Returns:
            Tuple of (DataFrame, data_hash)
        """
        logger.info(f"Loading data from {self.data_path}")
        
        # Read CSV
        df = pd.read_csv(self.data_path, low_memory=False)
        
        # Compute hash for data provenance
        data_hash = self._compute_hash()
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Data hash: {data_hash}")
        
        return df, data_hash
    
    def _compute_hash(self) -> str:
        """Compute SHA256 hash of the data file for versioning."""
        sha256_hash = hashlib.sha256()
        with open(self.data_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def get_basic_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the loaded data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with basic statistics
        """
        info = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }
        return info
