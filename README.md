# AMR Surveillance ML System

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Project Vision

Turn AMR water-fish-human surveillance data into a reproducible, validated Machine Learning system that:
- (a) Reproduces PDF findings
- (b) Discovers new AMR patterns via unsupervised learning
- (c) Predicts MDR and resistance with supervised learning
- (d) Delivers an operational dashboard and API for stakeholders

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [API Usage](#api-usage)
- [Dashboard](#dashboard)
- [Documentation](#documentation)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

### Data Processing
- ✅ Automated data cleaning and validation
- ✅ Standardized antibiotic encoding (S→0, I→1, R→2)
- ✅ MDR classification (resistant to ≥3 antibiotic classes)
- ✅ MAR index calculation and validation
- ✅ Missing value handling with documented policies

### Machine Learning
- ✅ **Unsupervised Learning**: PCA, UMAP, t-SNE, K-Means, Hierarchical Clustering, DBSCAN
- ✅ **Supervised Learning**: 6 algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, SVM)
- ✅ Model leaderboard and automated selection
- ✅ SHAP-based model interpretation
- ✅ Cross-validation with stratified K-fold

### Deployment
- ✅ RESTful API with FastAPI
- ✅ Interactive dashboard with Streamlit
- ✅ Docker containerization
- ✅ CI/CD with GitHub Actions
- ✅ Model versioning and tracking

### Monitoring & Security
- ✅ Performance monitoring
- ✅ Data drift detection
- ✅ Authentication and authorization
- ✅ Audit logging
- ✅ Security scanning (Bandit, Safety)

## Project Structure

```
thesis-project01/
├── data/
│   ├── raw/                    # Original untouched data
│   ├── interim/                # Intermediate processing steps
│   └── processed/              # Final cleaned data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_descriptive_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_supervised_learning.ipynb
├── src/
│   ├── data/                   # Data processing modules
│   ├── features/               # Feature engineering
│   ├── models/                 # Model training and evaluation
│   ├── api/                    # FastAPI application
│   └── dashboard/              # Streamlit dashboard
├── models/                     # Trained model artifacts
├── artifacts/                  # Generated artifacts (CSVs, plots)
├── deliverables/               # Final reports and documentation
├── tests/                      # Unit and integration tests
├── docs/                       # Documentation
├── config/                     # Configuration files
├── .github/workflows/          # CI/CD pipelines
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- (Optional) Docker for containerized deployment

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/Reyn4ldo/thesis-project01.git
cd thesis-project01
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
```bash
cp config/config.example.yml config/config.yml
# Edit config.yml with your settings
```

## Quick Start

### 1. Data Processing
```bash
# Run data cleaning pipeline
python src/data/clean_data.py --input data/raw/amr_surveillance_data.csv --output data/processed/cleaned_data.csv

# Validate processed data
python src/data/validate_data.py --input data/processed/cleaned_data.csv
```

### 2. Train Models
```bash
# Train all 6 supervised models
python src/models/train_models.py --config config/training_config.yml

# Or train specific model
python src/models/train_models.py --model xgboost
```

### 3. Start API
```bash
# Development mode
uvicorn src.api.main:app --reload --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Access API documentation at: http://localhost:8000/docs

### 4. Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

Access dashboard at: http://localhost:8501

## Data Processing

### Antibiotic Encoding Rules
- **S (Susceptible)** → 0
- **I (Intermediate)** → 1
- **R (Resistant)** → 2

### MDR Classification
An isolate is classified as **Multi-Drug Resistant (MDR)** if resistant to ≥3 antibiotic classes.

### MAR Index
```
MAR Index = (Number of antibiotics resistant) / (Number of antibiotics tested)
```

### Missing Value Policy
- Drop rows with >20% missing key antibiotic fields
- Impute remaining missing values using documented methods
- All transformations logged and reproducible

## Model Training

### Supervised Learning Pipeline

Six algorithms are trained and compared:
1. **Logistic Regression** (L1/L2 regularization)
2. **Decision Tree**
3. **Random Forest**
4. **XGBoost**
5. **LightGBM**
6. **Support Vector Machine**

### Evaluation Metrics
- ROC-AUC
- PR-AUC
- F1 Score
- Recall (MDR detection priority)
- Balanced Accuracy
- Calibration metrics

### Model Selection
Models are ranked on a leaderboard based on:
- Primary: ROC-AUC and MDR recall
- Secondary: Calibration and generalization across sites
- Tertiary: Computational efficiency

## API Usage

### Example: Predict MDR

```python
import requests

# Single isolate prediction
data = {
    "bacterial_species": "escherichia_coli",
    "sample_source": "drinking_water",
    "region": "region_iii_central_luzon",
    "ampicillin_int": "r",
    "amoxicillin_clavulanic_acid_int": "s",
    # ... other antibiotic results
}

response = requests.post("http://localhost:8000/predict_mdr", json=data)
result = response.json()

print(f"MDR Probability: {result['mdr_probability']:.2%}")
print(f"Classification: {result['classification']}")
print(f"Top risk factors: {result['top_features']}")
```

### Example: Upload and Process CSV

```python
files = {"file": open("new_data.csv", "rb")}
response = requests.post("http://localhost:8000/upload_csv", files=files)
summary = response.json()
```

## Dashboard

The Streamlit dashboard provides:
- **Upload Page**: Upload CSV data and view summary statistics
- **Descriptive Analysis**: Reproduce PDF findings with interactive visualizations
- **Clustering View**: Explore AMR patterns with dimensionality reduction plots
- **Prediction Tool**: Get MDR predictions with SHAP explanations
- **Report Generator**: Export comprehensive PDF reports

## Documentation

Detailed documentation available in `docs/`:
- [Data Dictionary](docs/data_dictionary.md)
- [Model Card](docs/model_card.md)
- [API Documentation](docs/api_documentation.md)
- [User Manual](docs/user_manual.md)
- [Developer Guide](docs/developer_guide.md)
- [Security Policy](docs/security_policy.md)
- [Maintenance SOP](docs/maintenance_sop.md)

## Development

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
pylint src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/
safety check
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test suite
pytest tests/test_data_processing.py
pytest tests/test_models.py
pytest tests/test_api.py
```

## Deployment

### Docker Deployment

```bash
# Build containers
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -n amr-surveillance

# View logs
kubectl logs -f deployment/api -n amr-surveillance
```

## CI/CD

GitHub Actions workflows:
- **Lint & Test**: Runs on every push and PR
- **Build Docker**: Builds container images
- **Deploy to Staging**: Auto-deploy on merge to main
- **Deploy to Production**: Manual approval required
- **Security Scan**: Daily vulnerability checks
- **Model Retraining**: Scheduled monthly

## Monitoring

- **Application Metrics**: Prometheus + Grafana
- **Model Performance**: MLflow tracking
- **Data Drift**: Great Expectations + custom monitors
- **Logs**: Centralized logging with structured logs
- **Alerts**: Slack/Email notifications for critical events

## Retraining Policy

Models are retrained:
- **Scheduled**: Quarterly (every 3 months)
- **Drift-triggered**: When data drift exceeds threshold
- **Performance-triggered**: When test performance drops >5%
- **Manual**: Upon domain expert request

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- All tests pass
- Code coverage >80%
- Code follows style guidelines
- Documentation is updated

## Team & Roles

- **Project Lead**: Domain oversight, stakeholder liaison
- **Data Engineer**: ETL pipelines, data quality
- **Data Scientist**: ML modeling, feature engineering
- **Backend Engineer**: API development, database integration
- **Frontend Engineer**: Dashboard UI/UX
- **DevOps Engineer**: CI/CD, containerization, monitoring
- **Domain Expert**: Model validation, clinical interpretation
- **Security Lead**: Access control, compliance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{amr_surveillance_ml,
  title={AMR Surveillance ML System},
  author={[Your Name]},
  year={2025},
  url={https://github.com/Reyn4ldo/thesis-project01}
}
```

## Acknowledgments

- Domain experts for validation and guidance
- Open-source ML community for tools and frameworks
- Local government units for data contribution

## Contact

For questions or support:
- **Email**: [your-email@example.com]
- **Issues**: [GitHub Issues](https://github.com/Reyn4ldo/thesis-project01/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Reyn4ldo/thesis-project01/discussions)

---

**Status**: 🚧 Active Development | **Version**: 0.1.0 | **Last Updated**: December 2025
