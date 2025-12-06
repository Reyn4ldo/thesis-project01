"""Tests for data loading functionality."""

import pandas as pd
import pytest
from pathlib import Path

from src.data.load_data import DataLoader


def test_data_loader_initialization():
    """Test DataLoader initialization with valid path."""
    data_path = Path(__file__).parent.parent / "data" / "raw" / "amr_surveillance_data.csv"
    
    if data_path.exists():
        loader = DataLoader(str(data_path))
        assert loader.data_path.exists()


def test_data_loader_invalid_path():
    """Test DataLoader with invalid path raises error."""
    with pytest.raises(FileNotFoundError):
        DataLoader("nonexistent_file.csv")


def test_load_data():
    """Test data loading returns DataFrame and hash."""
    data_path = Path(__file__).parent.parent / "data" / "raw" / "amr_surveillance_data.csv"
    
    if data_path.exists():
        loader = DataLoader(str(data_path))
        df, data_hash = loader.load()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(data_hash) == 64  # SHA256 hash length
