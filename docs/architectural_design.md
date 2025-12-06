# Architectural Design - AMR Surveillance ML System

## 1. Overview

This document describes the architectural design of the Antimicrobial Resistance (AMR) Surveillance Machine Learning System. The system transforms raw bacterial surveillance data into actionable MDR predictions through a scalable, maintainable, and secure architecture.

## 2. System Architecture

### 2.1 Architecture Style
- **Pattern**: Microservices-oriented with API-first design
- **Deployment**: Containerized services (Docker)
- **Communication**: RESTful HTTP/JSON
- **Data Flow**: Pipeline-based ETL with model inference

### 2.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Layer                                │
│  ┌──────────────────┐              ┌──────────────────┐        │
│  │   Web Browser    │              │   API Client     │        │
│  │   (Dashboard)    │              │   (curl/Python)  │        │
│  └────────┬─────────┘              └─────────┬────────┘        │
└───────────┼───────────────────────────────────┼─────────────────┘
            │                                   │
            │ HTTP/WebSocket                    │ HTTP/JSON
            │                                   │
┌───────────┼───────────────────────────────────┼─────────────────┐
│           │        Application Layer          │                 │
│  ┌────────▼─────────┐              ┌─────────▼────────┐        │
│  │   Streamlit      │              │   FastAPI        │        │
│  │   Dashboard      │              │   API Service    │        │
│  │   (Port 8501)    │              │   (Port 8000)    │        │
│  └────────┬─────────┘              └─────────┬────────┘        │
└───────────┼───────────────────────────────────┼─────────────────┘
            │                                   │
            └───────────────┬───────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                     Business Logic Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │     Data     │  │   Feature    │  │    Model     │         │
│  │  Processing  │  │ Engineering  │  │  Inference   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────────┐
│                        Data Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Raw Data   │  │  Processed   │  │   Trained    │         │
│  │    (CSV)     │  │   Data (CSV) │  │ Models (.pkl)│         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

## 3. Component Architecture

### 3.1 Data Processing Layer

#### 3.1.1 Data Ingestion Module
**Location**: `src/data/clean_data.py`

**Responsibilities**:
- Load raw CSV data
- Validate data schema
- Calculate provenance hash (SHA256)
- Handle file I/O errors

**Key Functions**:
- `load_raw_data(filepath)`: Load CSV with validation
- `validate_schema(df)`: Check required columns
- `calculate_hash(filepath)`: Generate SHA256 hash

**Input**: `data/raw/amr_surveillance_data.csv`  
**Output**: Pandas DataFrame

#### 3.1.2 Data Cleaning Module
**Location**: `src/data/clean_data.py`

**Responsibilities**:
- Remove duplicates
- Handle missing values (>20% threshold)
- Standardize antibiotic encodings (S→0, I→1, R→2)
- Calculate MDR classification
- Compute MAR index
- Generate cleaning metadata

**Key Functions**:
- `clean_data(df)`: Main cleaning pipeline
- `encode_antibiotics(df)`: Convert S/I/R to numeric
- `calculate_mdr(df)`: Classify MDR status
- `calculate_mar_index(df)`: Compute MAR values

**Input**: Raw DataFrame  
**Output**: 
- Cleaned DataFrame (487 rows)
- Metadata JSON (provenance, statistics)

**Design Decisions**:
- **20% threshold**: Balance data retention vs. quality
- **Ordinal encoding**: Preserve resistance severity
- **MDR definition**: ≥3 classes (WHO standard)

#### 3.1.3 Data Validation Module
**Location**: `src/data/validate_data.py`

**Responsibilities**:
- Validate cleaned data quality
- Check statistical properties
- Detect outliers and anomalies
- Generate validation report

**Key Functions**:
- `validate_processed_data(df)`: Run all validations
- `check_value_ranges(df)`: Verify numeric ranges
- `check_class_distribution(df)`: Validate MDR prevalence

**Output**: Validation report (pass/fail with details)

### 3.2 Feature Engineering Layer

#### 3.2.1 Feature Engineering Module
**Location**: `src/features/feature_engineering.py`

**Responsibilities**:
- Create engineered features
- Encode categorical variables
- Scale/normalize features
- Select relevant features

**Key Classes**:
- `FeatureEngineer`: Main feature engineering pipeline
  - `fit()`: Learn encodings/scalings from training data
  - `transform()`: Apply to new data
  - `fit_transform()`: Combined fit and transform

**Engineered Features** (29 total):
1. **Antibiotic Resistance** (17 features): Encoded S/I/R values
2. **Species Encoding** (1-hot): 13 species types
3. **Source Encoding** (1-hot): 4 source types
4. **Region Encoding** (1-hot): 4 region types
5. **Resistance Counts**: Per-class counts
6. **MAR Index**: Continuous [0, 1]

**Design Patterns**:
- Scikit-learn compatible (fit/transform API)
- Serializable (pickle)
- Reusable across training/inference

### 3.3 Model Training Layer

#### 3.3.1 Model Trainer Module
**Location**: `src/models/train_models.py`

**Responsibilities**:
- Train multiple ML models
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Model selection

**Supported Algorithms** (6):
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM
6. Support Vector Machine

**Key Functions**:
- `train_all_models()`: Train complete model suite
- `train_single_model(algorithm)`: Train specific model
- `evaluate_model(model, X_test, y_test)`: Compute metrics
- `select_best_model(models)`: Choose optimal model

**Training Pipeline**:
```python
1. Load processed data
2. Split train/test (80/20 stratified)
3. For each algorithm:
   a. Initialize model with hyperparameters
   b. Apply SMOTE for class balancing
   c. Train with 5-fold CV
   d. Evaluate on test set
   e. Save model artifact
4. Generate leaderboard
5. Select best model (composite score)
6. Save best model and metadata
```

**Output Artifacts**:
- Individual model files: `models/{algorithm}.pkl`
- Feature engineer: `models/feature_engineer.pkl`
- Leaderboard: `models/leaderboard.csv`
- Metadata: `models/training_metadata.json`

#### 3.3.2 Model Evaluation Module
**Location**: `src/models/evaluate_models.py`

**Responsibilities**:
- Calculate performance metrics
- Generate evaluation reports
- Create visualizations (ROC, PR curves)
- Compare models

**Key Metrics**:
- ROC-AUC: 0.9944 (Random Forest)
- Recall: 1.0000 (100% MDR detection)
- Balanced Accuracy: 0.9611
- Precision: 0.5333
- F1-Score: 0.6957

**Evaluation Functions**:
- `calculate_metrics(y_true, y_pred, y_proba)`: All metrics
- `plot_roc_curve(model, X_test, y_test)`: ROC visualization
- `plot_pr_curve(model, X_test, y_test)`: Precision-Recall
- `generate_leaderboard(models)`: Comparison table

#### 3.3.3 Model Predictor Module
**Location**: `src/models/predict.py`

**Responsibilities**:
- Load trained models
- Make predictions on new data
- Calculate prediction confidence
- Handle edge cases

**Key Functions**:
- `load_model(model_path)`: Load serialized model
- `predict_mdr(features)`: Single prediction
- `predict_batch(features_df)`: Batch predictions
- `get_feature_importance()`: Explain predictions

**Prediction Pipeline**:
```python
1. Load feature engineer
2. Load trained model
3. Transform input features
4. Generate prediction
5. Calculate probability
6. Return result with confidence
```

### 3.4 API Service Layer

#### 3.4.1 FastAPI Application
**Location**: `src/api/main.py`

**Architecture Pattern**: RESTful API with async/await

**Endpoints** (5):

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/health` | GET | Health check | None | Status, version, uptime |
| `/predict` | POST | Single MDR prediction | JSON isolate data | MDR probability, class |
| `/predict/batch` | POST | Batch predictions | JSON array | Array of predictions |
| `/upload` | POST | CSV file processing | Multipart file | Processing summary |
| `/models` | GET | Model information | None | Leaderboard, metadata |

**Key Components**:
- **Request Models** (Pydantic schemas):
  - `IsolateInput`: Single isolate validation
  - `BatchInput`: Batch request validation
  
- **Response Models**:
  - `PredictionResponse`: Prediction output
  - `HealthResponse`: Health check output
  - `ModelInfoResponse`: Model metadata

- **Middleware**:
  - CORS: Cross-origin requests (development)
  - Logging: Request/response logging
  - Error Handling: Standardized error responses

**API Design Principles**:
- **Validation**: Pydantic schemas for type safety
- **Documentation**: Auto-generated OpenAPI/Swagger
- **Async**: Non-blocking I/O for scalability
- **Versioning**: URL-based versioning (future: /v1/)
- **Error Handling**: Consistent error format

#### 3.4.2 API Router
**Location**: `src/api/routes.py`

**Responsibilities**:
- Route definitions
- Request handling
- Response formatting
- Error handling

**Routing Structure**:
```python
/health          → get_health_status()
/predict         → predict_mdr()
/predict/batch   → predict_batch()
/upload          → upload_csv()
/models          → get_model_info()
```

#### 3.4.3 API Models
**Location**: `src/api/models.py`

**Purpose**: Pydantic models for request/response validation

**Key Models**:
```python
class IsolateInput(BaseModel):
    bacterial_species: str
    sample_source: str
    region: str
    # 17 antibiotic fields (amp_int, amox_int, etc.)

class PredictionResponse(BaseModel):
    mdr_probability: float
    classification: str  # "MDR" or "Non-MDR"
    confidence: float
    top_features: List[str]

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    model_loaded: bool
```

### 3.5 Dashboard Layer

#### 3.5.1 Streamlit Application
**Location**: `src/dashboard/app.py`

**Architecture**: Multi-page Streamlit application

**Pages** (5):

1. **Home Page**:
   - Project overview
   - Navigation menu
   - Quick stats

2. **Data Upload & Analysis**:
   - CSV file upload
   - Data preview
   - Summary statistics
   - Distribution visualizations

3. **MDR Prediction**:
   - Interactive form (bacterial species, antibiotics)
   - Prediction button
   - Result display (gauge chart)
   - Confidence score

4. **Model Performance**:
   - Leaderboard table
   - Metric comparisons
   - Performance visualizations

5. **About**:
   - System information
   - Methodology summary
   - Contact information

**Key Features**:
- **Interactivity**: Forms, buttons, sliders
- **Visualizations**: Plotly charts (bar, gauge, scatter)
- **File I/O**: Upload CSV, download results
- **State Management**: Session state for multi-page

**Dashboard Design Principles**:
- **User-Friendly**: No coding required
- **Responsive**: Adapts to screen size
- **Real-Time**: Instant predictions
- **Visual**: Charts and gauges
- **Informative**: Help text and tooltips

## 4. Data Architecture

### 4.1 Data Storage

#### 4.1.1 File System Structure
```
data/
├── raw/                          # Original data (immutable)
│   └── amr_surveillance_data.csv  (583 rows)
├── interim/                      # Intermediate processing
│   └── (temporary files)
└── processed/                    # Clean data (analysis-ready)
    ├── cleaned_data.csv           (487 rows)
    └── cleaning_metadata.json     (audit trail)

models/
├── random_forest.pkl             # Best model (176KB)
├── xgboost.pkl                   # Alternative models
├── lightgbm.pkl
├── logistic_regression.pkl
├── decision_tree.pkl
├── svm.pkl
├── feature_engineer.pkl          # Feature transformer (3KB)
├── leaderboard.csv               # Model comparison
└── training_metadata.json        # Training details
```

#### 4.1.2 Data Schemas

**Raw Data Schema** (CSV):
- `Isolate_ID`: Unique identifier
- `Bacterial_Species`: Species name
- `Sample_Source`: Water/fish/human
- `Region`: Geographic region
- `Ampicillin_int`: S/I/R
- ... (17 antibiotic fields)
- `Site`: Collection site

**Processed Data Schema** (CSV):
- All raw fields (cleaned)
- `is_mdr`: Boolean MDR classification
- `mar_index`: Float [0, 1]
- Encoded antibiotics (0/1/2)

**Model Input Schema** (29 features):
- 17 antibiotic resistance values (0/1/2)
- 13 species one-hot features
- 4 source one-hot features
- 4 region one-hot features
- 1 MAR index

### 4.2 Data Flow

#### 4.2.1 Training Data Flow
```
Raw CSV (583 rows)
    ↓
[Data Cleaning]
    ↓
Cleaned CSV (487 rows)
    ↓
[Feature Engineering]
    ↓
Feature Matrix (487 × 29)
    ↓
[Train/Test Split 80/20]
    ↓
Training Set (389 × 29) + Test Set (98 × 29)
    ↓
[Model Training (6 algorithms)]
    ↓
Trained Models (.pkl files)
    ↓
[Model Evaluation]
    ↓
Leaderboard + Best Model Selection
```

#### 4.2.2 Prediction Data Flow
```
User Input (Web/API)
    ↓
[Input Validation]
    ↓
Raw Features (dict/JSON)
    ↓
[Feature Engineering Transform]
    ↓
Feature Vector (1 × 29)
    ↓
[Model Inference]
    ↓
Prediction Probability (0-1)
    ↓
[Threshold Classification (0.5)]
    ↓
MDR/Non-MDR + Confidence
    ↓
User Output (JSON/UI)
```

## 5. Deployment Architecture

### 5.1 Containerization

#### 5.1.1 Docker Architecture

**API Container**:
- **Base Image**: `python:3.10-slim`
- **Application**: FastAPI + Uvicorn
- **Port**: 8000
- **Health Check**: `/health` endpoint (30s interval)
- **Size**: ~500MB

**Dashboard Container**:
- **Base Image**: `python:3.10-slim`
- **Application**: Streamlit
- **Port**: 8501
- **Health Check**: HTTP GET to port 8501
- **Size**: ~500MB

**Docker Compose**:
```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports: ["8000:8000"]
    volumes: ["./data:/app/data", "./models:/app/models"]
    
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports: ["8501:8501"]
    depends_on: [api]
    environment: ["API_URL=http://api:8000"]
```

#### 5.1.2 Container Orchestration
- **Local**: Docker Compose
- **Production**: Kubernetes (future)
- **Service Discovery**: Internal DNS
- **Load Balancing**: Reverse proxy (Nginx)

### 5.2 CI/CD Pipeline

#### 5.2.1 GitHub Actions Workflow

**Trigger**: Push to main, Pull Request

**Stages**:

1. **Test** (5 min):
   - Install dependencies
   - Run pytest
   - Generate coverage report
   - Fail if coverage <80%

2. **Lint** (2 min):
   - Black (code formatting)
   - Flake8 (style guide)
   - isort (import sorting)
   - mypy (type checking)

3. **Security** (3 min):
   - Bandit (security linting)
   - Safety (dependency vulnerabilities)
   - Report issues as warnings

4. **Build** (5 min):
   - Build Docker images
   - Tag with commit SHA
   - Push to container registry

5. **Deploy** (2 min):
   - Deploy to staging (auto)
   - Deploy to production (manual approval)
   - Health check validation

**Total Pipeline Time**: ~15 minutes

#### 5.2.2 Deployment Environments

| Environment | Trigger | Approval | URL |
|-------------|---------|----------|-----|
| **Development** | Every commit | Auto | localhost |
| **Staging** | Merge to main | Auto | staging.example.com |
| **Production** | Manual | Required | prod.example.com |

### 5.3 Scalability Architecture

#### 5.3.1 Horizontal Scaling
- **API**: Multiple replicas behind load balancer
- **Dashboard**: Stateless, multiple instances
- **Models**: Shared volume (NFS/S3)

#### 5.3.2 Performance Optimization
- **Model Loading**: Lazy loading, singleton pattern
- **Caching**: In-memory cache for predictions
- **Async I/O**: Non-blocking API requests
- **Connection Pooling**: Database connections (future)

#### 5.3.3 Capacity Planning
- **Current**: 100 concurrent users
- **Target**: 1000 concurrent users
- **Scaling Strategy**: Horizontal (add more containers)

## 6. Security Architecture

### 6.1 Security Layers

#### 6.1.1 Application Security
- **Input Validation**: Pydantic schemas
- **SQL Injection**: N/A (no database yet)
- **XSS Prevention**: Content-Type headers
- **CSRF Protection**: Token-based (future)

#### 6.1.2 API Security
- **Authentication**: API keys (future)
- **Authorization**: Role-based access (future)
- **Rate Limiting**: 100 req/min per IP (future)
- **CORS**: Restricted origins in production

#### 6.1.3 Data Security
- **Encryption in Transit**: HTTPS/TLS
- **Encryption at Rest**: Volume encryption (future)
- **Data Anonymization**: No PII in datasets
- **Access Control**: File system permissions

#### 6.1.4 Infrastructure Security
- **Container Scanning**: Trivy/Clair
- **Dependency Scanning**: Safety, Dependabot
- **Secrets Management**: Environment variables
- **Network Segmentation**: Private networks

### 6.2 Security Monitoring
- **Audit Logging**: All API requests logged
- **Intrusion Detection**: Log analysis (future)
- **Vulnerability Scanning**: Weekly automated scans
- **Incident Response**: Documented procedures

## 7. Monitoring & Observability

### 7.1 Logging Architecture

#### 7.1.1 Log Levels
- **DEBUG**: Detailed diagnostic info
- **INFO**: General informational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages (recoverable)
- **CRITICAL**: Critical failures (unrecoverable)

#### 7.1.2 Log Aggregation
- **Format**: Structured JSON logs
- **Collection**: Centralized log server (future)
- **Retention**: 30 days (application), 90 days (audit)

### 7.2 Metrics Architecture

#### 7.2.1 Application Metrics
- **Request Rate**: Requests per second
- **Response Time**: P50, P95, P99 latency
- **Error Rate**: 4xx, 5xx errors
- **Throughput**: MB/s processed

#### 7.2.2 Model Metrics
- **Prediction Volume**: Predictions per day
- **Prediction Distribution**: MDR vs. Non-MDR ratio
- **Confidence Scores**: Average confidence
- **Model Version**: Currently deployed model

#### 7.2.3 Infrastructure Metrics
- **CPU Usage**: Per container
- **Memory Usage**: Per container
- **Disk I/O**: Read/write IOPS
- **Network I/O**: Bytes in/out

### 7.3 Alerting
- **Health Check Failures**: Immediate alert
- **High Error Rate**: >5% in 5 minutes
- **High Latency**: P95 >1 second
- **Model Drift**: Distribution change detected

## 8. Technology Stack

### 8.1 Programming Languages
- **Python 3.10+**: Primary language
- **YAML**: Configuration files
- **Markdown**: Documentation

### 8.2 Frameworks & Libraries

#### 8.2.1 Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: Preprocessing, pipelines

#### 8.2.2 Machine Learning
- **Scikit-learn**: Traditional ML algorithms
- **XGBoost**: Gradient boosting
- **LightGBM**: Fast gradient boosting
- **Imbalanced-learn**: SMOTE for class balancing

#### 8.2.3 API & Web
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **Streamlit**: Dashboard framework

#### 8.2.4 Visualization
- **Plotly**: Interactive charts
- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations

#### 8.2.5 DevOps
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **pytest**: Unit testing
- **GitHub Actions**: CI/CD

#### 8.2.6 Code Quality
- **Black**: Code formatting
- **Flake8**: Linting
- **isort**: Import sorting
- **mypy**: Type checking
- **Bandit**: Security linting

### 8.3 Infrastructure
- **Version Control**: Git + GitHub
- **Container Registry**: Docker Hub / GitHub Container Registry
- **Cloud Provider**: Flexible (AWS/GCP/Azure)
- **Orchestration**: Docker Compose (current), Kubernetes (future)

## 9. Design Patterns

### 9.1 Software Design Patterns

#### 9.1.1 Singleton Pattern
- **Usage**: Model loading (load once, reuse)
- **Benefit**: Reduced memory, faster predictions

#### 9.1.2 Factory Pattern
- **Usage**: Model instantiation (create models by algorithm name)
- **Benefit**: Extensible, maintainable

#### 9.1.3 Pipeline Pattern
- **Usage**: Data processing (sequential transformations)
- **Benefit**: Composable, testable

#### 9.1.4 Repository Pattern
- **Usage**: Data access (abstract file I/O)
- **Benefit**: Testable, swappable storage

### 9.2 Architectural Patterns

#### 9.2.1 Separation of Concerns
- **Data Layer**: Storage and retrieval
- **Business Logic**: Processing and modeling
- **Presentation Layer**: API and UI
- **Benefit**: Modularity, testability

#### 9.2.2 API-First Design
- **API as contract**: Defined before implementation
- **Benefit**: Parallel development, clear interfaces

#### 9.2.3 Immutable Infrastructure
- **Containers**: Immutable, versioned images
- **Benefit**: Reproducibility, rollback capability

## 10. Future Architecture Enhancements

### 10.1 Database Integration
- **Current**: File-based storage
- **Future**: PostgreSQL for predictions, metadata
- **Benefit**: ACID transactions, query performance

### 10.2 Message Queue
- **Technology**: RabbitMQ / Redis
- **Use Case**: Async batch predictions, job queues
- **Benefit**: Decoupling, scalability

### 10.3 Caching Layer
- **Technology**: Redis
- **Use Case**: Cache frequent predictions
- **Benefit**: Reduced latency, lower load

### 10.4 Kubernetes Deployment
- **Features**: Auto-scaling, self-healing, rolling updates
- **Benefit**: Production-grade orchestration

### 10.5 Model Registry
- **Technology**: MLflow
- **Use Case**: Model versioning, experiment tracking
- **Benefit**: Better model governance

## 11. Conclusion

This architectural design provides a robust, scalable, and maintainable foundation for the AMR Surveillance ML System. The modular design allows for independent development and testing of components, while the containerized deployment ensures reproducibility and portability.

**Key Architectural Strengths**:
- **Modularity**: Clear separation of concerns
- **Scalability**: Horizontal scaling capable
- **Maintainability**: Well-documented, tested code
- **Security**: Multiple security layers
- **Observability**: Comprehensive monitoring
- **Reproducibility**: Containerized, version-controlled

**System Characteristics**:
- **Performance**: <100ms API response time
- **Availability**: 99.9% uptime target
- **Scalability**: 100+ concurrent users
- **Reliability**: Automated testing, health checks
- **Maintainability**: Modular, documented, tested

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Status**: Final  
**Maintainer**: Architecture Team
