# Research Methodology - AMR Surveillance ML System

## 1. Overview

This document describes the research methodology employed in developing a Machine Learning-based Antimicrobial Resistance (AMR) surveillance system for analyzing bacterial isolates from water, fish, and human sources.

## 2. Research Design

### 2.1 Research Type
- **Type**: Applied Research with Experimental Design
- **Approach**: Quantitative Analysis using Machine Learning
- **Paradigm**: Data-Driven Model Development
- **Objective**: Develop and validate ML models for MDR prediction

### 2.2 Research Framework
The methodology follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

## 3. Data Collection

### 3.1 Data Source
- **Source**: AMR surveillance data from water, fish, and human samples
- **Collection Period**: Historical surveillance data (specific timeframe documented in data provenance)
- **Geographic Coverage**: Philippines (multiple regions)
- **Sample Types**: 
  - Drinking water
  - Aquaculture samples
  - Fish samples
  - Human samples

### 3.2 Data Characteristics
- **Total Records**: 583 bacterial isolates (raw data)
- **Variables**: 
  - Sample metadata (species, source, region)
  - Antibiotic susceptibility test results
  - Resistance patterns
- **Format**: CSV (Comma-Separated Values)
- **Quality**: Real-world surveillance data with inherent noise

### 3.3 Data Provenance
- **Tracking**: SHA256 hash-based provenance
- **Versioning**: Git-based version control
- **Audit Trail**: Complete data lineage documentation

## 4. Data Processing Pipeline

### 4.1 Data Cleaning
**Objective**: Transform raw surveillance data into analysis-ready format

#### 4.1.1 Missing Value Handling
- **Policy**: Drop rows with >20% missing antibiotic fields
- **Imputation**: Document all transformations
- **Result**: 487 clean isolates from 583 raw (83.5% retention)

#### 4.1.2 Data Validation
- **Schema Validation**: Ensure all required fields present
- **Range Validation**: Check antibiotic test results (S, I, R)
- **Consistency Checks**: Verify MDR classifications
- **Outlier Detection**: Identify and document anomalies

#### 4.1.3 Data Transformation
**Antibiotic Encoding**:
- Susceptible (S) → 0
- Intermediate (I) → 1
- Resistant (R) → 2

**Rationale**: Ordinal encoding preserves resistance severity information

### 4.2 Feature Engineering
**Objective**: Create meaningful features for ML modeling

#### 4.2.1 Engineered Features
1. **MDR Classification**: Binary target variable
   - MDR = 1 if resistant to ≥3 antibiotic classes
   - MDR = 0 otherwise
   - Prevalence: 8.0% (39/487 isolates)

2. **MAR Index**: Multiple Antibiotic Resistance Index
   - Formula: MAR = (Number of antibiotics resistant) / (Number tested)
   - Range: [0, 1]
   - Mean: 0.103

3. **Resistance Counts**: 
   - Per-class resistance counts
   - Total resistance count
   - Resistance ratios

4. **Categorical Encodings**:
   - Species: One-hot encoding
   - Source: One-hot encoding
   - Region: One-hot encoding

#### 4.2.2 Feature Selection
- **Initial Features**: 50+ raw features
- **Selected Features**: 29 engineered features
- **Selection Criteria**:
  - Domain relevance
  - Low multicollinearity
  - Statistical significance
  - Model interpretability

### 4.3 Data Splitting
- **Training Set**: 80% (389 samples)
- **Test Set**: 20% (98 samples)
- **Stratification**: Preserve MDR class distribution
- **Cross-Validation**: 5-fold stratified CV

## 5. Machine Learning Methodology

### 5.1 Problem Formulation
**Primary Task**: Binary Classification (MDR vs. Non-MDR)

**Challenges**:
- Class imbalance (92% non-MDR, 8% MDR)
- High dimensionality
- Interpretability requirements
- Clinical significance of false negatives

### 5.2 Algorithm Selection
**Rationale**: Compare diverse algorithm families for optimal performance

#### 5.2.1 Algorithms Evaluated (6 Models)
1. **Logistic Regression**
   - Linear baseline model
   - L1/L2 regularization
   - Interpretable coefficients

2. **Decision Tree**
   - Non-linear, rule-based
   - Feature importance via splits
   - Interpretable tree structure

3. **Random Forest**
   - Ensemble of decision trees
   - Robust to overfitting
   - Feature importance ranking

4. **XGBoost**
   - Gradient boosting
   - State-of-the-art performance
   - Handles imbalance well

5. **LightGBM**
   - Fast gradient boosting
   - Efficient for large datasets
   - Low memory footprint

6. **Support Vector Machine (SVM)**
   - Maximum margin classifier
   - RBF kernel for non-linearity
   - Robust to outliers

### 5.3 Model Training

#### 5.3.1 Training Protocol
- **Cross-Validation**: 5-fold stratified K-fold
- **Hyperparameter Tuning**: Grid search with CV
- **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Training Time**: ~90 seconds (all 6 models)

#### 5.3.2 Hyperparameters
Each model optimized for:
- Regularization strength
- Tree depth/complexity
- Learning rate
- Number of estimators
- Kernel parameters (SVM)

### 5.4 Model Evaluation

#### 5.4.1 Evaluation Metrics
**Primary Metrics**:
- **ROC-AUC**: Overall discrimination ability
- **Recall**: MDR detection sensitivity (critical for surveillance)
- **Balanced Accuracy**: Handles class imbalance

**Secondary Metrics**:
- Precision
- F1-Score
- PR-AUC (Precision-Recall)
- Calibration metrics

#### 5.4.2 Model Selection Criteria
**Composite Score Formula**:
```
Score = 0.4 × ROC-AUC + 0.3 × Recall + 0.3 × Balanced Accuracy
```

**Rationale**:
- ROC-AUC (40%): Overall performance
- Recall (30%): Critical for MDR detection
- Balanced Accuracy (30%): Class imbalance handling

**Selection Process**:
1. Train all 6 models
2. Evaluate on test set
3. Calculate composite scores
4. Select highest-scoring model
5. Validate on holdout data

#### 5.4.3 Model Performance Results

| Model | ROC-AUC | Recall | Balanced Acc. | Score |
|-------|---------|--------|---------------|-------|
| **Random Forest** | **0.9944** | **1.0000** | **0.9611** | **0.9574** |
| XGBoost | 1.0000 | 0.8750 | 0.9326 | 0.9568 |
| LightGBM | 1.0000 | 0.8750 | 0.9326 | 0.9568 |
| Logistic Regression | 0.9990 | 0.8750 | 0.9326 | 0.9521 |
| SVM | 0.9990 | 0.8750 | 0.9326 | 0.9521 |
| Decision Tree | 0.9333 | 0.8750 | 0.9196 | 0.9221 |

**Winner**: Random Forest (highest composite score)

**Selection Rationale**: While XGBoost and LightGBM achieved perfect ROC-AUC (1.0000), Random Forest was selected because:
1. **Perfect Recall** (1.0000 vs. 0.8750): Critical for MDR surveillance - no false negatives
2. **Better Balanced Accuracy** (0.9611 vs. 0.9326): Superior handling of class imbalance
3. **Highest Composite Score** (0.9574): Best overall performance across all weighted metrics
4. **Robustness**: Less prone to overfitting compared to boosting methods on this dataset

### 5.5 Model Interpretation

#### 5.5.1 Feature Importance
- Extract feature importance from Random Forest
- Identify top predictors of MDR
- Validate with domain knowledge

#### 5.5.2 Explainability
- SHAP (SHapley Additive exPlanations) values
- Feature contribution visualization
- Clinical interpretation of predictions

## 6. Validation Strategy

### 6.1 Internal Validation
- **K-Fold Cross-Validation**: 5-fold stratified
- **Holdout Test Set**: 20% unseen data
- **Bootstrap Validation**: 1000 iterations

### 6.2 External Validation
- **Temporal Validation**: Test on future data (when available)
- **Geographic Validation**: Test on different regions
- **Source Validation**: Test across sample types

### 6.3 Robustness Testing
- **Sensitivity Analysis**: Vary training parameters
- **Stability Testing**: Multiple random seeds
- **Edge Case Testing**: Extreme values, missing data

## 7. Deployment Methodology

### 7.1 API Development
- **Framework**: FastAPI (modern, async)
- **Endpoints**: 5 RESTful endpoints
- **Validation**: Pydantic schemas
- **Documentation**: Auto-generated Swagger/OpenAPI

### 7.2 Dashboard Development
- **Framework**: Streamlit (rapid prototyping)
- **Pages**: 5 interactive pages
- **Visualizations**: Plotly (interactive charts)
- **User Experience**: Intuitive, no-code interface

### 7.3 Containerization
- **Technology**: Docker
- **Services**: API, Dashboard (separate containers)
- **Orchestration**: Docker Compose
- **Benefits**: Reproducibility, portability, scalability

### 7.4 CI/CD Pipeline
**GitHub Actions Workflow**:
1. **Test**: Run unit tests, integration tests
2. **Lint**: Code quality checks (Black, Flake8, isort)
3. **Security**: Vulnerability scanning (Bandit, Safety)
4. **Build**: Docker image creation
5. **Deploy**: Automated deployment to staging/production

## 8. Monitoring & Maintenance

### 8.1 Performance Monitoring
- **Metrics**: Response time, throughput, error rates
- **Health Checks**: Every 30 seconds
- **Logging**: Structured logs for debugging

### 8.2 Model Monitoring
- **Data Drift Detection**: Monitor input distribution changes
- **Performance Degradation**: Track model metrics over time
- **Retraining Triggers**: Automated alerts for model updates

### 8.3 Retraining Protocol
**Schedule**:
- Quarterly (every 3 months)
- Performance-triggered (>5% drop)
- Data drift-triggered (distribution shift)
- Manual (domain expert request)

**Process**:
1. Collect new labeled data (minimum 100 isolates)
2. Validate data quality
3. Retrain all 6 models
4. Evaluate and compare to current model
5. Deploy if improved, otherwise investigate
6. Document retraining decision

## 9. Ethical Considerations

### 9.1 Data Privacy
- **Anonymization**: No personal identifiers
- **Access Control**: Role-based permissions
- **Compliance**: GDPR, local regulations

### 9.2 Model Fairness
- **Bias Detection**: Test across demographic groups
- **Equity**: Ensure equal performance across sources/regions
- **Transparency**: Document limitations

### 9.3 Clinical Safety
- **False Negative Risk**: Prioritize recall (100% achieved)
- **Human Oversight**: Predictions support, not replace, clinicians
- **Validation**: Domain expert review of predictions

## 10. Quality Assurance

### 10.1 Code Quality
- **Standards**: PEP 8 (Python)
- **Type Checking**: MyPy static analysis
- **Code Review**: Peer review process
- **Testing**: >80% code coverage target

### 10.2 Documentation Quality
- **Completeness**: All modules documented
- **Clarity**: User-friendly language
- **Maintenance**: Updated with code changes
- **Formats**: Markdown, inline docstrings

### 10.3 Reproducibility
- **Version Control**: Git for all code/configs
- **Dependency Management**: requirements.txt
- **Seed Management**: Fixed random seeds
- **Environment**: Docker containers

## 11. Limitations

### 11.1 Data Limitations
- **Sample Size**: 487 isolates (moderate)
- **Class Imbalance**: 8% MDR prevalence
- **Geographic Scope**: Philippines only
- **Temporal Scope**: Single time period

### 11.2 Model Limitations
- **Generalization**: May not generalize to other regions/pathogens
- **Feature Dependency**: Requires complete antibiotic test results
- **Interpretability**: Ensemble models less interpretable than linear
- **Static Model**: Requires retraining for concept drift

### 11.3 Deployment Limitations
- **Infrastructure**: Requires server/cloud resources
- **Connectivity**: API requires internet access
- **Latency**: Network-dependent response times
- **Scalability**: Current setup for moderate load

## 12. Future Directions

### 12.1 Model Enhancements
- Deep learning approaches (neural networks)
- Multi-task learning (predict specific antibiotics)
- Transfer learning from other AMR datasets
- Ensemble meta-models

### 12.2 Data Enhancements
- Increase sample size (target: 1000+ isolates)
- Temporal analysis (time series patterns)
- External datasets for validation
- Genomic data integration (if available)

### 12.3 System Enhancements
- Real-time predictions
- Mobile application
- Multi-language support
- Advanced visualization (geographic maps)

## 13. Conclusion

This methodology provides a rigorous, reproducible approach to developing an ML-based AMR surveillance system. The combination of robust data processing, comprehensive model evaluation, and production-ready deployment ensures that the system meets both scientific standards and operational requirements.

**Key Strengths**:
- Systematic, documented approach
- Multiple algorithm comparison
- Emphasis on recall for public health
- Production-ready infrastructure
- Comprehensive validation

**Deliverables**:
- High-performance ML models (99.4% ROC-AUC)
- Operational API and dashboard
- Complete documentation
- CI/CD pipeline
- Monitoring and maintenance protocols

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Status**: Final  
**Maintainer**: Project Team
