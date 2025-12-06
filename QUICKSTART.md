# Quick Start Guide - AMR Surveillance ML System

## 🚀 Get Started in 5 Minutes

### Option 1: Run with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Reyn4ldo/thesis-project01.git
cd thesis-project01

# Build and start services
docker-compose up -d

# Access the services
# Dashboard: http://localhost:8501
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Run Locally

```bash
# Clone and setup
git clone https://github.com/Reyn4ldo/thesis-project01.git
cd thesis-project01

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Process data
python src/data/run_cleaning.py \
  --input data/raw/amr_surveillance_data.csv \
  --output data/processed/cleaned_data.csv

# Train models
python src/models/run_training.py \
  --input data/processed/cleaned_data.csv \
  --output-dir models

# Start API
uvicorn src.api.main:app --reload

# In another terminal, start dashboard
streamlit run src/dashboard/app.py
```

## 📝 Quick Examples

### Example 1: Process Data

```bash
python src/data/run_cleaning.py \
  --input data/raw/amr_surveillance_data.csv \
  --output data/processed/cleaned_data.csv \
  --verbose
```

**Output**: Cleaned CSV with MDR classification

### Example 2: Train Models

```bash
python src/models/run_training.py \
  --input data/processed/cleaned_data.csv \
  --output-dir models \
  --cv-folds 5
```

**Output**: 6 trained models + leaderboard

### Example 3: API Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bacterial_species": "escherichia_coli",
    "sample_source": "drinking_water",
    "administrative_region": "region_iii_central_luzon",
    "ampicillin_int": "r",
    "tetracycline_int": "r",
    "gentamicin_int": "s"
  }'
```

**Response**:
```json
{
  "mdr_probability": 0.75,
  "mdr_classification": "MDR",
  "confidence": "High"
}
```

## 🎯 Common Tasks

### Task 1: Upload and Analyze Data

1. Open dashboard: http://localhost:8501
2. Go to "Data Upload & Analysis"
3. Upload your CSV file
4. Click "Clean and Process Data"
5. Download cleaned results

### Task 2: Get MDR Prediction

1. Open dashboard: http://localhost:8501
2. Go to "MDR Prediction"
3. Fill in isolate information
4. Enter antibiotic results (S/I/R)
5. Click "Predict MDR"
6. Review probability and classification

### Task 3: View Model Performance

1. Open dashboard: http://localhost:8501
2. Go to "Model Performance"
3. View leaderboard and metrics
4. Compare model performance

## 📊 Expected Results

### Data Processing
- **Input**: 583 raw isolates
- **Output**: 487 clean isolates (96 dropped due to missing values)
- **MDR Identified**: 39 isolates (8.0% prevalence)

### Model Training
- **Models Trained**: 6 algorithms
- **Best Model**: Random Forest
- **Performance**: 99.4% ROC-AUC, 100% MDR recall

### API Performance
- **Response Time**: <100ms typical
- **Availability**: 99.9% uptime target
- **Capacity**: 100+ requests/second

## 🔧 Troubleshooting

### Issue: Import errors

**Solution**:
```bash
# Ensure you're in project root and virtual environment is activated
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Models not found

**Solution**:
```bash
# Train models first
python src/models/run_training.py \
  --input data/processed/cleaned_data.csv \
  --output-dir models
```

### Issue: Dashboard won't start

**Solution**:
```bash
# Install streamlit
pip install streamlit

# Run with explicit path
streamlit run src/dashboard/app.py --server.port=8501
```

## 📚 Next Steps

1. **Read the User Manual**: `docs/user_manual.md`
2. **Review Model Card**: `docs/model_card.md`
3. **Check Data Dictionary**: `docs/data_dictionary.md`
4. **Explore API**: http://localhost:8000/docs
5. **Run Tests**: `pytest tests/`

## 🆘 Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: GitHub Issues
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: Built-in help and tooltips

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Data cleaning runs without errors
- [ ] Models train successfully
- [ ] API responds at http://localhost:8000/health
- [ ] Dashboard loads at http://localhost:8501
- [ ] Can make predictions via API
- [ ] Can upload CSV to dashboard
- [ ] Model leaderboard displays correctly

---

**Quick Start Version**: 1.0  
**For**: AMR Surveillance ML System v0.1.0  
**Last Updated**: December 2025
