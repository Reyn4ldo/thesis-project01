# User Manual - AMR Surveillance ML System

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Using the Dashboard](#using-the-dashboard)
4. [Using the API](#using-the-api)
5. [Data Requirements](#data-requirements)
6. [Interpreting Results](#interpreting-results)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

## Introduction

### What is the AMR Surveillance ML System?

The AMR Surveillance ML System is a machine learning-powered platform for predicting Multi-Drug Resistance (MDR) in bacterial isolates from environmental and food surveillance programs. It helps public health officials, microbiologists, and researchers identify high-risk isolates and monitor AMR trends.

### Key Features

- 🔬 **Automated Data Processing**: Upload CSV files and automatically clean and validate data
- 🎯 **MDR Prediction**: Predict MDR status with 99.4% accuracy using machine learning
- 📊 **Interactive Dashboard**: Visualize AMR trends and patterns
- 🚀 **REST API**: Integrate predictions into existing workflows
- 📈 **Performance Tracking**: Monitor model performance over time

### What is MDR?

**Multi-Drug Resistance (MDR)** means a bacterial isolate is resistant to antibiotics from **3 or more** different antibiotic classes. MDR bacteria pose significant public health risks as treatment options are limited.

## Getting Started

### System Requirements

- **Web Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **Internet Connection**: Required for accessing the dashboard and API
- **Data Format**: CSV files with antibiotic susceptibility test results

### Access Methods

#### Method 1: Web Dashboard (Easiest)
1. Open your web browser
2. Navigate to: `http://[your-server-address]:8501`
3. No installation required!

#### Method 2: REST API (For Integration)
- **Base URL**: `http://[your-server-address]:8000`
- **Documentation**: `http://[your-server-address]:8000/docs`

#### Method 3: Local Installation
See the [README.md](../README.md) for detailed installation instructions.

## Using the Dashboard

### 1. Home Page

The home page provides an overview of the system's capabilities:
- Data analysis features
- MDR prediction capabilities
- Model performance metrics

**Navigation**: Use the sidebar to switch between different pages.

### 2. Data Upload & Analysis

#### Uploading Data

1. Click **"Data Upload & Analysis"** in the sidebar
2. Click **"Browse files"** and select your CSV file
3. Wait for the file to upload
4. Review the data preview

#### Processing Data

1. Click the **"Clean and Process Data"** button
2. Wait for processing to complete (usually 5-10 seconds)
3. Review the processing summary:
   - Total isolates processed
   - Number of MDR isolates detected
   - MDR rate percentage
   - Mean MAR index

#### Viewing Visualizations

After processing, the system automatically generates:
- **Species Distribution**: Bar chart showing bacterial species counts
- **Sample Source Distribution**: Pie chart of sample sources
- **MDR Rate by Source**: Bar chart comparing MDR rates across sources

#### Downloading Results

Click **"Download Cleaned CSV"** to save the processed data to your computer.

### 3. MDR Prediction

#### Single Isolate Prediction

1. Click **"MDR Prediction"** in the sidebar
2. Fill in the isolate information:
   - **Bacterial Species**: Select from dropdown
   - **Sample Source**: Water, fish, etc.
   - **Administrative Region**: Geographic location
3. Enter antibiotic results:
   - Select **S** (Susceptible), **I** (Intermediate), or **R** (Resistant)
   - Leave blank if not tested
4. Click **"Predict MDR"** button
5. Review results:
   - **Classification**: MDR or Non-MDR
   - **Probability**: Likelihood of MDR (0-100%)
   - **Confidence**: High, Medium, or Low

#### Understanding Predictions

**High Confidence** (Probability >80% or <20%)
- Strong evidence for the prediction
- Can be acted upon with confidence

**Medium Confidence** (Probability 60-80% or 20-40%)
- Moderate evidence
- Consider additional testing

**Low Confidence** (Probability 40-60%)
- Unclear prediction
- Recommend confirmatory testing

### 4. Model Performance

View detailed performance metrics for all trained models:
- **Leaderboard**: Ranking of all 6 models
- **Performance Metrics**: Accuracy, ROC-AUC, Recall, F1-Score
- **Model Comparison**: Visual comparison charts

**Key Metric**: ROC-AUC of 0.994 means the model correctly distinguishes MDR from non-MDR in 99.4% of cases.

## Using the API

### API Documentation

Complete interactive API documentation is available at:
```
http://[your-server-address]:8000/docs
```

### Common API Endpoints

#### 1. Health Check
```bash
curl http://[server]:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "models_loaded": true
}
```

#### 2. Single Prediction
```bash
curl -X POST http://[server]:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bacterial_species": "escherichia_coli",
    "sample_source": "drinking_water",
    "administrative_region": "region_iii_central_luzon",
    "ampicillin_int": "r",
    "tetracycline_int": "r"
  }'
```

**Response**:
```json
{
  "mdr_probability": 0.85,
  "mdr_classification": "MDR",
  "confidence": "High"
}
```

#### 3. Upload CSV
```bash
curl -X POST http://[server]:8000/upload \
  -F "file=@your_data.csv"
```

#### 4. List Models
```bash
curl http://[server]:8000/models
```

## Data Requirements

### CSV File Format

Your CSV file must contain the following columns:

#### Required Metadata
- `bacterial_species`: Bacterial species name
- `sample_source`: Source of sample
- `administrative_region`: Geographic region

#### Required Antibiotic Columns

At least some of these columns (with `_int` suffix):
- `ampicillin_int`
- `amoxicillin_clavulanic_acid_int`
- `cefalotin_int`
- `gentamicin_int`
- `enrofloxacin_int`
- `tetracycline_int`
- `chloramphenicol_int`
- And others...

#### Antibiotic Values
- Use **S**, **I**, or **R** (case-insensitive)
- Special markers (*, ≤, ≥) are automatically cleaned
- Missing values are acceptable (up to 20% per isolate)

### Example CSV

```csv
bacterial_species,sample_source,administrative_region,ampicillin_int,tetracycline_int
escherichia_coli,drinking_water,region_iii_central_luzon,R,R
klebsiella_pneumoniae,river_water,region_iii_central_luzon,S,I
```

### Data Quality Tips

✅ **Do**:
- Use consistent naming conventions
- Include all available antibiotic results
- Verify species names are correct
- Check for typos in column names

❌ **Don't**:
- Mix different encoding schemes (e.g., 0/1/2 and S/I/R)
- Include personally identifiable information
- Use non-standard antibiotic names

## Interpreting Results

### MDR Classification

**MDR (Multi-Drug Resistant)**
- ⚠️ Resistant to ≥3 antibiotic classes
- **Action**: Enhanced surveillance, infection control measures
- **Significance**: Limited treatment options

**Non-MDR**
- ✅ Resistant to <3 antibiotic classes
- **Action**: Standard surveillance
- **Significance**: Multiple treatment options available

### MAR Index

The **Multiple Antibiotic Resistance (MAR) Index** ranges from 0 to 1:
- **0.0**: Susceptible to all tested antibiotics
- **0.2-0.5**: Moderate resistance
- **>0.5**: High resistance burden
- **1.0**: Resistant to all tested antibiotics

**Interpretation**:
- MAR > 0.2 suggests exposure to high antibiotic pressure
- Useful for comparing resistance levels across samples

### Confidence Levels

| Confidence | Probability Range | Interpretation |
|------------|------------------|----------------|
| **High** | >80% or <20% | Strong prediction, act with confidence |
| **Medium** | 60-80% or 20-40% | Moderate certainty, consider context |
| **Low** | 40-60% | Unclear, recommend confirmatory testing |

## Troubleshooting

### Common Issues

#### Dashboard Won't Load
**Problem**: Browser shows error or blank page  
**Solutions**:
1. Check internet connection
2. Verify server is running
3. Try a different browser
4. Clear browser cache

#### CSV Upload Fails
**Problem**: Error when uploading file  
**Solutions**:
1. Check file format (must be CSV)
2. Verify column names match requirements
3. Ensure file size < 10MB
4. Check for special characters in data

#### Prediction Error
**Problem**: Error message after clicking "Predict"  
**Solutions**:
1. Ensure at least 5-6 antibiotics are entered
2. Check that values are S, I, or R only
3. Verify species and source are selected

#### Models Not Loaded
**Problem**: "Models not loaded" error  
**Solutions**:
1. Contact system administrator
2. Verify models directory exists
3. Check server logs for errors

### Getting Help

If you encounter issues not covered here:
1. Check the [FAQ](#faq) section below
2. Contact your system administrator
3. Review server logs (if accessible)
4. Create an issue on GitHub (if applicable)

## FAQ

### General Questions

**Q: Is this system for clinical use?**  
A: No. This is a surveillance tool for monitoring AMR trends. It should not be used for clinical diagnosis or treatment decisions.

**Q: How accurate is the prediction?**  
A: The model achieves 99.4% ROC-AUC and 100% recall for MDR detection on test data. However, always confirm critical findings with traditional methods.

**Q: Can I use this offline?**  
A: The web dashboard requires a server connection. For offline use, you would need to install the system locally (see Developer Guide).

### Data Questions

**Q: What if I'm missing some antibiotic results?**  
A: The system can handle up to 20% missing values per isolate. More than that, and the isolate may be excluded from analysis.

**Q: Can I upload data from other countries?**  
A: Yes, but note that the model was trained on Philippines data. Validation on local data is recommended before operational use.

**Q: What format should dates be in?**  
A: Dates are not required for prediction. If included for tracking, use ISO format (YYYY-MM-DD).

### Prediction Questions

**Q: What does "composite score" mean?**  
A: It's a weighted combination of performance metrics (40% ROC-AUC, 30% Recall, 30% Balanced Accuracy) used to rank models.

**Q: Why are there false positives?**  
A: The model prioritizes sensitivity (finding all MDR cases) over precision. Some non-MDR isolates may be flagged as MDR, which is acceptable for surveillance.

**Q: Can I adjust the prediction threshold?**  
A: Currently, the threshold is fixed at 0.5 (50% probability). Contact the development team if you need a custom threshold for your use case.

### Technical Questions

**Q: What technology stack is used?**  
A: Python, scikit-learn, FastAPI, Streamlit, Docker. See the Model Card for full details.

**Q: Is the API rate-limited?**  
A: Currently no rate limits. For high-volume use, contact administrators to ensure adequate capacity.

**Q: Can I download the trained model?**  
A: Model files are available in the `models/` directory if you have server access. For external use, please contact the project team.

---

## Quick Reference Card

### Dashboard Pages
| Page | Purpose |
|------|---------|
| Home | Overview and introduction |
| Data Upload | Process and visualize surveillance data |
| MDR Prediction | Predict individual isolate MDR status |
| Model Performance | View model metrics and comparisons |
| About | System information and details |

### Key Metrics
| Metric | Value | Meaning |
|--------|-------|---------|
| ROC-AUC | 0.994 | Overall discrimination ability |
| Recall | 1.000 | % of MDR cases detected |
| Precision | 0.533 | % of predicted MDR that are truly MDR |

### Support Contacts
- **Technical Support**: [IT Contact]
- **Scientific Questions**: [Microbiology Contact]
- **Training Requests**: [Training Coordinator]

---

**User Manual Version**: 1.0  
**Last Updated**: December 2025  
**For**: AMR Surveillance ML System v0.1.0
