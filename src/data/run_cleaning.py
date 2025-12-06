#!/usr/bin/env python
"""
Command-line interface for data cleaning pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.clean_data import DataCleaner
from src.data.load_data import DataLoader


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    """Main entry point for data cleaning."""
    parser = argparse.ArgumentParser(
        description="Clean AMR surveillance data"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw CSV data file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save cleaned CSV data"
    )
    parser.add_argument(
        "--treat-intermediate-as-resistant",
        action="store_true",
        help="Treat intermediate (I) as resistant for MDR calculation"
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.20,
        help="Drop rows with more than this fraction of missing antibiotics (default: 0.20)"
    )
    parser.add_argument(
        "--impute-method",
        type=str,
        choices=["mode", "median", "drop"],
        default="mode",
        help="Method for imputing missing values (default: mode)"
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
        # Load data
        logger.info("=" * 60)
        logger.info("AMR Data Cleaning Pipeline")
        logger.info("=" * 60)
        
        loader = DataLoader(args.input)
        df_raw, data_hash = loader.load()
        
        logger.info(f"Data provenance hash: {data_hash}")
        
        # Get basic info
        info = loader.get_basic_info(df_raw)
        logger.info(f"Loaded {info['n_rows']} rows, {info['n_columns']} columns")
        
        # Clean data
        cleaner = DataCleaner(
            treat_intermediate_as_resistant=args.treat_intermediate_as_resistant,
            missing_threshold=args.missing_threshold,
            impute_method=args.impute_method
        )
        
        df_clean = cleaner.clean(df_raw)
        
        # Save cleaned data
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cleaner.save_cleaned_data(df_clean, args.output, save_metadata=True)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Cleaning Summary")
        logger.info("=" * 60)
        logger.info(f"Input rows: {len(df_raw)}")
        logger.info(f"Output rows: {len(df_clean)}")
        logger.info(f"Rows dropped: {len(df_raw) - len(df_clean)}")
        
        if "is_mdr" in df_clean.columns:
            mdr_count = df_clean["is_mdr"].sum()
            logger.info(f"MDR isolates: {mdr_count} ({mdr_count/len(df_clean):.1%})")
        
        if "mar_index_calculated" in df_clean.columns:
            mar_mean = df_clean["mar_index_calculated"].mean()
            logger.info(f"Mean MAR index: {mar_mean:.4f}")
        
        logger.info("=" * 60)
        logger.info("✓ Data cleaning completed successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
