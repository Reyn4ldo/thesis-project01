# Model Card - AMR Surveillance MDR Prediction System

## Model Details

### Overview
- **Model Name**: AMR MDR Prediction System
- **Model Version**: 1.0.0
- **Model Date**: December 2025
- **Model Type**: Binary Classification (Supervised Learning)
- **Primary Algorithm**: Random Forest (selected from 6-model ensemble)

### Developers
- **Organization**: AMR Surveillance Project Team
- **Contact**: [Project Contact Information]

### Model Description
This model predicts whether a bacterial isolate exhibits Multi-Drug Resistance (MDR) based on antimicrobial susceptibility testing (AST) results. The model was trained on water-fish-human surveillance data from the Philippines to support public health surveillance and intervention planning.

## Intended Use

### Primary Use Cases
1. **Surveillance**: Monitor MDR trends across different sample sources (water, fish)
2. **Risk Assessment**: Identify high-risk isolates requiring enhanced infection control
3. **Research**: Support epidemiological studies of AMR patterns
4. **Public Health**: Inform policy decisions and intervention strategies

### Intended Users
- Microbiologists and laboratory staff
- Epidemiologists and public health officials
- Research scientists studying AMR
- Local government units (LGUs) and health departments

### Out-of-Scope Uses
- **Not for clinical diagnosis**: This model is for surveillance purposes only
- **Not a replacement for AST**: Traditional susceptibility testing is still required
- **Not for treatment decisions**: Clinical decisions should be made by qualified healthcare professionals
- **Limited geographic generalizability**: Trained on Philippines data; may not generalize to other regions without validation

## Training Data

### Data Source
- **Origin**: Water-fish-human AMR surveillance program, Philippines
- **Time Period**: [Data collection period]
- **Geographic Coverage**: Region III (Central Luzon) and surrounding areas
- **Sample Sources**: Drinking water, river water, fish (tilapia, bangus)

### Dataset Characteristics
- **Total Isolates**: 487 (after cleaning from 583 raw samples)
- **MDR Prevalence**: 8.0% (39 MDR / 448 non-MDR)
- **Bacterial Species**: 
  - Escherichia coli
  - Klebsiella pneumoniae subspecies pneumoniae
  - Enterobacter cloacae complex
  - Enterobacter aerogenes

### Features
- **Total Features**: 29
- **Antibiotic Interpretations**: 23 antibiotics (S/I/R encoded as 0/1/2)
- **Metadata**: Bacterial species, sample source, administrative region
- **Derived Features**: Number resistant, number tested, MAR index

### Data Preprocessing
1. **Encoding**: S→0, I→1, R→2
2. **Missing Value Handling**: Drop rows with >20% missing antibiotics
3. **MDR Definition**: Resistant to ≥3 antibiotic classes
4. **Train-Test Split**: 80/20 stratified split
5. **Feature Scaling**: StandardScaler applied to all numeric features

## Model Architecture

### Algorithm Selection Process
Six algorithms were trained and compared:
1. Logistic Regression (L1/L2 regularization)
2. Decision Tree
3. Random Forest ⭐ **SELECTED**
4. XGBoost
5. LightGBM
6. Support Vector Machine

### Selection Criteria
Random Forest was selected based on composite score:
- **Composite Score Formula**: 0.4 × ROC-AUC + 0.3 × Recall + 0.3 × Balanced Accuracy
- **Rationale**: Balances overall discrimination (ROC-AUC), MDR detection sensitivity (Recall), and class balance (Balanced Accuracy)

### Hyperparameters (Random Forest)
```python
{
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42
}
```

### Training Configuration
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Class Balancing**: Balanced class weights to address 8% MDR prevalence
- **Evaluation Metrics**: Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

## Performance

### Test Set Performance (Random Forest)
| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.9944 |
| **Recall (MDR Detection)** | 1.0000 |
| **Precision** | 0.5333 |
| **F1-Score** | 0.6957 |
| **Balanced Accuracy** | 0.9611 |
| **Overall Accuracy** | 0.9286 |

### Cross-Validation Performance
| Metric | Mean CV Score |
|--------|---------------|
| **ROC-AUC** | 0.9831 |
| **Recall** | 0.9333 |
| **F1-Score** | 0.7980 |
| **Balanced Accuracy** | 0.9472 |

### Confusion Matrix (Test Set)
```
                Predicted
                Non-MDR  MDR
Actual Non-MDR    83      7
       MDR         0      8
```

### Key Strengths
✅ **Perfect MDR Detection**: 100% recall on test set (no false negatives)  
✅ **Excellent Discrimination**: 99.4% ROC-AUC indicates strong ability to separate classes  
✅ **Balanced Performance**: High balanced accuracy (96.1%) despite class imbalance  
✅ **Robust**: Consistent performance across CV folds

### Limitations
⚠️ **Moderate Precision**: 53.3% precision means some false positives (acceptable for surveillance)  
⚠️ **Limited Sample Size**: 487 isolates may limit generalizability  
⚠️ **Class Imbalance**: Only 8% MDR may affect model calibration  
⚠️ **Geographic Specificity**: Trained on Philippines data; may not generalize globally

## Ethical Considerations

### Fairness and Bias
- **Surveillance Context**: Model is used for population-level surveillance, not individual treatment
- **No Protected Attributes**: Model uses only microbiological data (species, antibiotics), no human demographic data
- **Geographic Bias**: Trained on Philippines data; performance in other regions unknown

### Privacy
- **No Personal Data**: Dataset contains only bacterial isolate information
- **De-identified**: No patient or location-identifying information included
- **Aggregated Reporting**: Results should be reported at population level

### Potential Harms
- **False Negatives**: Missing MDR isolates could delay public health response (mitigated by 100% recall)
- **False Positives**: Over-alerting may cause resource misallocation (acceptable trade-off for surveillance)
- **Misuse**: Model should not be used for clinical treatment decisions

### Benefits
- **Early Warning**: Rapid MDR detection supports timely public health interventions
- **Resource Optimization**: Focuses attention on high-risk isolates and locations
- **Surveillance Enhancement**: Complements traditional laboratory methods

## Caveats and Recommendations

### When to Use This Model
✅ Surveillance of AMR trends in environmental and food samples  
✅ Risk stratification for further investigation  
✅ Research on AMR patterns and drivers  
✅ Rapid screening before confirmatory testing

### When NOT to Use This Model
❌ Clinical diagnosis or treatment decisions  
❌ As sole determinant for infection control measures  
❌ Outside the Philippines without local validation  
❌ For bacterial species not in training data

### Recommended Practices
1. **Complement Traditional Methods**: Use alongside standard AST, not as replacement
2. **Continuous Monitoring**: Track model performance over time
3. **Regular Retraining**: Update model quarterly or when performance degrades
4. **Expert Review**: Have domain experts validate predictions on critical cases
5. **Local Validation**: Validate on local data before deployment in new regions

## Model Maintenance

### Monitoring
- **Performance Metrics**: Track ROC-AUC, recall, calibration monthly
- **Data Drift**: Monitor input distribution changes (species, resistance patterns)
- **Concept Drift**: Watch for changes in MDR prevalence or resistance mechanisms

### Retraining Policy
**Scheduled**: Quarterly retraining with new surveillance data  
**Triggered**: Retrain if any of the following occur:
- Test set ROC-AUC drops below 0.95
- MDR recall drops below 0.95
- Significant change in species or resistance distribution
- Introduction of new antibiotics or testing methods

### Versioning
- **Model Versioning**: Semantic versioning (MAJOR.MINOR.PATCH)
- **Data Versioning**: SHA256 hash of training data
- **Artifact Storage**: All models and artifacts stored with metadata
- **Rollback**: Previous stable version retained for emergency rollback

## References

### Scientific Basis
1. **MDR Definition**: Based on international consensus definitions for MDR in Gram-negative bacteria
2. **Antibiotic Classes**: CLSI and EUCAST guidelines for classification
3. **Machine Learning**: Standard practices for imbalanced classification problems

### Related Work
- WHO Global Antimicrobial Resistance and Use Surveillance System (GLASS)
- CLSI M100 Performance Standards for Antimicrobial Susceptibility Testing
- Machine learning applications in AMR prediction literature

## Contact Information

For questions, feedback, or to report issues:
- **Technical Issues**: [GitHub Issues Link]
- **Scientific Questions**: [Project Email]
- **Collaboration Inquiries**: [Contact Information]

## Changelog

### Version 1.0.0 (December 2025)
- Initial release
- 6-model comparison completed
- Random Forest selected as best model
- Achieved 99.4% test ROC-AUC, 100% MDR recall

---

**Model Card Version**: 1.0  
**Last Updated**: December 2025  
**Next Review**: March 2026
