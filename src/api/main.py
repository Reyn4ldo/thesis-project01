"""
FastAPI application for AMR surveillance system.
Provides REST API endpoints for MDR prediction and data processing.
"""

import sys
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    IsolateInput,
    PredictionResponse,
    UploadResponse,
)
from src.data.clean_data import DataCleaner
from src.features.engineer_features import FeatureEngineer

# Initialize FastAPI app
app = FastAPI(
    title="AMR Surveillance ML API",
    description="REST API for Antimicrobial Resistance surveillance and MDR prediction",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
# TODO: Configure allowed origins for production deployment
# For production, replace "*" with specific domains, e.g., ["https://yourdomain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # SECURITY: Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and feature engineer
MODELS = {}
FEATURE_ENGINEER = None
BEST_MODEL = None
BEST_MODEL_NAME = None


def load_models():
    """Load trained models and feature engineer at startup."""
    global MODELS, FEATURE_ENGINEER, BEST_MODEL, BEST_MODEL_NAME
    
    models_dir = Path(__file__).parent.parent.parent / "models"
    
    try:
        # Load feature engineer
        fe_path = models_dir / "feature_engineer.pkl"
        if fe_path.exists():
            FEATURE_ENGINEER = joblib.load(fe_path)
            print(f"✓ Loaded feature engineer from {fe_path}")
        
        # Load leaderboard to get best model
        leaderboard_path = models_dir / "leaderboard.csv"
        if leaderboard_path.exists():
            leaderboard = pd.read_csv(leaderboard_path, index_col=0)
            BEST_MODEL_NAME = leaderboard.index[0]
            print(f"✓ Best model identified: {BEST_MODEL_NAME}")
        
        # Load best model
        if BEST_MODEL_NAME:
            best_model_path = models_dir / f"{BEST_MODEL_NAME}.pkl"
            if best_model_path.exists():
                BEST_MODEL = joblib.load(best_model_path)
                print(f"✓ Loaded best model: {BEST_MODEL_NAME}")
        
        # Load all models
        for model_file in models_dir.glob("*.pkl"):
            if model_file.stem not in ["feature_engineer"]:
                try:
                    MODELS[model_file.stem] = joblib.load(model_file)
                    print(f"✓ Loaded model: {model_file.stem}")
                except Exception as e:
                    print(f"✗ Failed to load {model_file.stem}: {e}")
        
        print(f"✓ Total models loaded: {len(MODELS)}")
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    print("=" * 60)
    print("AMR Surveillance API - Starting Up")
    print("=" * 60)
    load_models()
    print("=" * 60)
    print("✓ API Ready")
    print("=" * 60)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AMR Surveillance ML API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=BEST_MODEL is not None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(isolate: IsolateInput):
    """
    Predict MDR for a single isolate.
    
    Args:
        isolate: Isolate data with antibiotic results
        
    Returns:
        MDR prediction with probability and confidence
    """
    if BEST_MODEL is None or FEATURE_ENGINEER is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert input to DataFrame
        isolate_dict = isolate.dict()
        df = pd.DataFrame([isolate_dict])
        
        # Clean and prepare features
        cleaner = DataCleaner()
        df_clean = cleaner.clean(df)
        
        # Engineer features
        X, _, _ = FEATURE_ENGINEER.prepare_for_modeling(
            df_clean, target_col=None, fit=False, scale=True
        )
        
        # Make prediction
        if hasattr(BEST_MODEL, "predict_proba"):
            mdr_prob = float(BEST_MODEL.predict_proba(X)[0, 1])
        else:
            mdr_prob = float(BEST_MODEL.predict(X)[0])
        
        # Determine classification and confidence
        classification = "MDR" if mdr_prob >= 0.5 else "Non-MDR"
        
        if mdr_prob >= 0.8 or mdr_prob <= 0.2:
            confidence = "High"
        elif mdr_prob >= 0.6 or mdr_prob <= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return PredictionResponse(
            mdr_probability=mdr_prob,
            mdr_classification=classification,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict MDR for multiple isolates.
    
    Args:
        request: Batch prediction request with list of isolates
        
    Returns:
        Batch predictions with summary statistics
    """
    if BEST_MODEL is None or FEATURE_ENGINEER is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        predictions = []
        
        for isolate in request.isolates:
            # Use single prediction endpoint
            pred = await predict_single(isolate)
            predictions.append(pred)
        
        # Calculate summary
        mdr_count = sum(1 for p in predictions if p.mdr_classification == "MDR")
        non_mdr_count = len(predictions) - mdr_count
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            mdr_count=mdr_count,
            non_mdr_count=non_mdr_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/upload", response_model=UploadResponse, tags=["Data"])
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload and process CSV file with AMR data.
    
    Args:
        file: CSV file with isolate data
        
    Returns:
        Processing summary with statistics
    """
    if BEST_MODEL is None or FEATURE_ENGINEER is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read CSV
        contents = await file.read()
        from io import StringIO
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        
        # Clean data
        cleaner = DataCleaner()
        df_clean = cleaner.clean(df)
        
        # Calculate statistics
        if "is_mdr" in df_clean.columns:
            mdr_count = int(df_clean["is_mdr"].sum())
            non_mdr_count = len(df_clean) - mdr_count
        else:
            mdr_count = 0
            non_mdr_count = len(df_clean)
        
        summary_stats = {
            "total_isolates": len(df_clean),
            "mdr_percentage": (mdr_count / len(df_clean) * 100) if len(df_clean) > 0 else 0,
            "mean_mar_index": float(df_clean["mar_index_calculated"].mean()) if "mar_index_calculated" in df_clean.columns else 0,
            "unique_species": int(df_clean["bacterial_species"].nunique()) if "bacterial_species" in df_clean.columns else 0
        }
        
        return UploadResponse(
            message="CSV processed successfully",
            rows_processed=len(df_clean),
            mdr_count=mdr_count,
            non_mdr_count=non_mdr_count,
            summary_stats=summary_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.get("/models", tags=["Models"])
async def list_models():
    """
    List all available models and their performance.
    
    Returns:
        Dictionary of models and metrics
    """
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        models_dir = Path(__file__).parent.parent.parent / "models"
        leaderboard_path = models_dir / "leaderboard.csv"
        
        if leaderboard_path.exists():
            leaderboard = pd.read_csv(leaderboard_path, index_col=0)
            models_info = {
                "best_model": BEST_MODEL_NAME,
                "available_models": list(MODELS.keys()),
                "leaderboard": leaderboard.to_dict(orient="index")
            }
            return models_info
        else:
            return {
                "best_model": BEST_MODEL_NAME,
                "available_models": list(MODELS.keys())
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
