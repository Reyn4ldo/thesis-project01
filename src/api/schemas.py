"""
Pydantic schemas for API request/response validation.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IsolateInput(BaseModel):
    """Input schema for single isolate prediction."""
    
    bacterial_species: str = Field(..., description="Bacterial species name")
    sample_source: str = Field(..., description="Sample source (e.g., drinking_water, fish_tilapia)")
    administrative_region: str = Field(..., description="Administrative region")
    
    # Antibiotic interpretations (S/I/R)
    ampicillin_int: Optional[str] = None
    amoxicillin_clavulanic_acid_int: Optional[str] = None
    ceftaroline_int: Optional[str] = None
    cefalexin_int: Optional[str] = None
    cefalotin_int: Optional[str] = None
    cefpodoxime_int: Optional[str] = None
    cefotaxime_int: Optional[str] = None
    cefovecin_int: Optional[str] = None
    ceftiofur_int: Optional[str] = None
    ceftazidime_avibactam_int: Optional[str] = None
    imepenem_int: Optional[str] = None
    amikacin_int: Optional[str] = None
    gentamicin_int: Optional[str] = None
    neomycin_int: Optional[str] = None
    nalidixic_acid_int: Optional[str] = None
    enrofloxacin_int: Optional[str] = None
    marbofloxacin_int: Optional[str] = None
    pradofloxacin_int: Optional[str] = None
    doxycycline_int: Optional[str] = None
    tetracycline_int: Optional[str] = None
    nitrofurantoin_int: Optional[str] = None
    chloramphenicol_int: Optional[str] = None
    trimethoprim_sulfamethazole_int: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "bacterial_species": "escherichia_coli",
                "sample_source": "drinking_water",
                "administrative_region": "region_iii_central_luzon",
                "ampicillin_int": "r",
                "amoxicillin_clavulanic_acid_int": "s",
                "cefalotin_int": "r",
                "gentamicin_int": "s",
                "enrofloxacin_int": "r",
                "tetracycline_int": "r"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for MDR prediction."""
    
    isolate_id: Optional[str] = None
    mdr_probability: float = Field(..., description="Probability of MDR (0-1)")
    mdr_classification: str = Field(..., description="Classification: MDR or Non-MDR")
    confidence: str = Field(..., description="Confidence level: High, Medium, Low")
    top_risk_features: Optional[List[Dict[str, float]]] = Field(
        None, description="Top features contributing to prediction"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "isolate_id": "sample_001",
                "mdr_probability": 0.85,
                "mdr_classification": "MDR",
                "confidence": "High",
                "top_risk_features": [
                    {"feature": "ampicillin_encoded", "contribution": 0.25},
                    {"feature": "tetracycline_encoded", "contribution": 0.18}
                ]
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    
    isolates: List[IsolateInput] = Field(..., description="List of isolates to predict")
    
    class Config:
        json_schema_extra = {
            "example": {
                "isolates": [
                    {
                        "bacterial_species": "escherichia_coli",
                        "sample_source": "drinking_water",
                        "administrative_region": "region_iii_central_luzon",
                        "ampicillin_int": "r",
                        "tetracycline_int": "r"
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[PredictionResponse]
    total: int
    mdr_count: int
    non_mdr_count: int


class UploadResponse(BaseModel):
    """Response schema for CSV upload."""
    
    message: str
    rows_processed: int
    mdr_count: int
    non_mdr_count: int
    summary_stats: Dict


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    models_loaded: bool
