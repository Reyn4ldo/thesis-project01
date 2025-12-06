"""
Data cleaning module for AMR surveillance data.
Implements standardized preprocessing pipeline following documented rules.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess AMR surveillance data."""
    
    # Antibiotic encoding mapping: S->0, I->1, R->2
    RESISTANCE_ENCODING = {"s": 0, "i": 1, "r": 2}
    
    # Antibiotic classes for MDR calculation
    ANTIBIOTIC_CLASSES = {
        "beta_lactams": [
            "ampicillin", "amoxicillin/clavulanic_acid", "ceftaroline",
            "cefalexin", "cefalotin", "cefpodoxime", "cefotaxime",
            "cefovecin", "ceftiofur", "ceftazidime/avibactam"
        ],
        "carbapenems": ["imepenem"],
        "aminoglycosides": ["amikacin", "gentamicin", "neomycin"],
        "fluoroquinolones": [
            "nalidixic_acid", "enrofloxacin", "marbofloxacin", "pradofloxacin"
        ],
        "tetracyclines": ["doxycycline", "tetracycline"],
        "nitrofurans": ["nitrofurantoin"],
        "phenicols": ["chloramphenicol"],
        "sulfonamides": ["trimethoprim/sulfamethazole"]
    }
    
    def __init__(
        self,
        treat_intermediate_as_resistant: bool = False,
        missing_threshold: float = 0.20,
        impute_method: str = "mode"
    ):
        """
        Initialize DataCleaner.
        
        Args:
            treat_intermediate_as_resistant: If True, treat 'I' as resistant for MDR
            missing_threshold: Drop rows with more than this fraction of missing antibiotics
            impute_method: Method for imputing missing values ('mode', 'median', 'drop')
        """
        self.treat_intermediate_as_resistant = treat_intermediate_as_resistant
        self.missing_threshold = missing_threshold
        self.impute_method = impute_method
        self.transformation_log = []
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute full cleaning pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline")
        df_clean = df.copy()
        
        # Step 1: Standardize column names
        df_clean = self._standardize_columns(df_clean)
        
        # Step 2: Extract and encode antibiotic interpretations
        df_clean = self._encode_antibiotics(df_clean)
        
        # Step 3: Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Step 4: Calculate MDR classification
        df_clean = self._calculate_mdr(df_clean)
        
        # Step 5: Validate and recalculate MAR index
        df_clean = self._calculate_mar_index(df_clean)
        
        # Step 6: Clean metadata columns
        df_clean = self._clean_metadata(df_clean)
        
        logger.info(f"Cleaning complete. Final shape: {df_clean.shape}")
        logger.info(f"Applied {len(self.transformation_log)} transformations")
        
        return df_clean
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistency."""
        df = df.copy()
        
        # Convert column names to lowercase and replace spaces/special chars
        df.columns = [
            col.lower().replace(" ", "_").replace("/", "_")
            for col in df.columns
        ]
        
        self.transformation_log.append("Standardized column names")
        return df
    
    def _encode_antibiotics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode antibiotic interpretation columns (S/I/R) to numeric (0/1/2).
        """
        df = df.copy()
        
        # Find all _int columns (interpretation columns)
        int_columns = [col for col in df.columns if col.endswith("_int")]
        
        for col in int_columns:
            if col in df.columns:
                # Clean the values: lowercase, strip, handle special markers
                df[col] = df[col].astype(str).str.lower().str.strip()
                
                # Remove special markers like * or <= that appear with S/I/R
                df[col] = df[col].str.replace(r"[^sir]", "", regex=True)
                
                # Replace empty strings with NaN
                df[col] = df[col].replace("", np.nan)
                df[col] = df[col].replace("nan", np.nan)
                
                # Create encoded column (numeric version)
                encoded_col = col.replace("_int", "_encoded")
                df[encoded_col] = df[col].map(self.RESISTANCE_ENCODING)
        
        self.transformation_log.append(
            f"Encoded {len(int_columns)} antibiotic interpretation columns"
        )
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to documented policy."""
        df = df.copy()
        initial_rows = len(df)
        
        # Get encoded antibiotic columns
        encoded_cols = [col for col in df.columns if col.endswith("_encoded")]
        
        # Calculate missingness per row
        missing_fraction = df[encoded_cols].isnull().sum(axis=1) / len(encoded_cols)
        
        # Drop rows exceeding missing threshold
        rows_to_drop = missing_fraction > self.missing_threshold
        df = df[~rows_to_drop]
        
        dropped_rows = initial_rows - len(df)
        logger.info(
            f"Dropped {dropped_rows} rows with >{self.missing_threshold:.0%} "
            f"missing antibiotic values"
        )
        
        # Impute remaining missing values
        if self.impute_method == "mode":
            for col in encoded_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                    df[col].fillna(mode_val, inplace=True)
        elif self.impute_method == "drop":
            df = df.dropna(subset=encoded_cols)
        
        self.transformation_log.append(
            f"Handled missing values: dropped {dropped_rows} rows, "
            f"imputed with {self.impute_method}"
        )
        return df
    
    def _calculate_mdr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MDR classification.
        MDR = resistant to ≥3 antibiotic classes
        """
        df = df.copy()
        
        # For each antibiotic class, check if resistant to at least one antibiotic
        resistant_threshold = 2 if not self.treat_intermediate_as_resistant else 1
        
        resistant_classes = []
        for class_name, antibiotics in self.ANTIBIOTIC_CLASSES.items():
            # Find encoded columns for this class
            class_cols = [
                col.replace("_int", "_encoded")
                for col in df.columns
                if any(ab in col for ab in antibiotics) and col.endswith("_encoded")
            ]
            
            if class_cols:
                # Check if resistant to any antibiotic in this class
                class_resistant = (
                    df[class_cols] >= resistant_threshold
                ).any(axis=1).astype(int)
                resistant_classes.append(class_resistant)
        
        # Sum resistant classes
        if resistant_classes:
            df["resistant_classes_count"] = pd.concat(
                resistant_classes, axis=1
            ).sum(axis=1)
            df["is_mdr"] = (df["resistant_classes_count"] >= 3).astype(int)
        else:
            df["resistant_classes_count"] = 0
            df["is_mdr"] = 0
        
        mdr_count = df["is_mdr"].sum()
        logger.info(f"Identified {mdr_count} MDR isolates ({mdr_count/len(df):.1%})")
        
        self.transformation_log.append(
            f"Calculated MDR classification: {mdr_count} MDR isolates"
        )
        return df
    
    def _calculate_mar_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MAR (Multiple Antibiotic Resistance) index.
        MAR = (# resistant) / (# tested)
        """
        df = df.copy()
        
        # Get all encoded columns
        encoded_cols = [col for col in df.columns if col.endswith("_encoded")]
        
        # Count resistant (value = 2)
        df["num_resistant"] = (df[encoded_cols] == 2).sum(axis=1)
        
        # Count tested (non-missing)
        df["num_tested"] = df[encoded_cols].notna().sum(axis=1)
        
        # Calculate MAR index
        df["mar_index_calculated"] = df["num_resistant"] / df["num_tested"]
        df["mar_index_calculated"] = df["mar_index_calculated"].fillna(0)
        
        # Compare with original if exists
        if "mar_index" in df.columns:
            df["mar_index_original"] = pd.to_numeric(
                df["mar_index"], errors="coerce"
            )
            # Check discrepancies
            discrepancy = (
                (df["mar_index_calculated"] - df["mar_index_original"]).abs() > 0.01
            )
            n_discrepancies = discrepancy.sum()
            if n_discrepancies > 0:
                logger.warning(
                    f"Found {n_discrepancies} discrepancies between original "
                    f"and calculated MAR index"
                )
        
        self.transformation_log.append("Calculated MAR index")
        return df
    
    def _clean_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean metadata columns (species, region, source, etc.)."""
        df = df.copy()
        
        # Standardize categorical columns
        categorical_cols = [
            "bacterial_species", "administrative_region", "national_site",
            "local_site", "sample_source", "esbl"
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].replace("nan", np.nan)
        
        # Convert numeric columns
        numeric_cols = ["replicate", "colony", "scored_resistance", "num_antibiotics_tested"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        self.transformation_log.append("Cleaned metadata columns")
        return df
    
    def get_transformation_log(self) -> List[str]:
        """Return log of all transformations applied."""
        return self.transformation_log
    
    def save_cleaned_data(
        self, df: pd.DataFrame, output_path: str, save_metadata: bool = True
    ) -> None:
        """
        Save cleaned data and optionally metadata.
        
        Args:
            df: Cleaned DataFrame
            output_path: Path to save CSV
            save_metadata: If True, save transformation log and statistics
        """
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")
        
        if save_metadata:
            import json
            from pathlib import Path
            
            metadata = {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": list(df.columns),
                "mdr_count": int(df["is_mdr"].sum()) if "is_mdr" in df.columns else 0,
                "transformation_log": self.transformation_log,
                "missing_threshold": self.missing_threshold,
                "treat_intermediate_as_resistant": self.treat_intermediate_as_resistant
            }
            
            metadata_path = Path(output_path).parent / "cleaning_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {metadata_path}")
