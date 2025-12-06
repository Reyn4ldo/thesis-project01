"""
Streamlit Dashboard for AMR Surveillance System.
Provides interactive UI for data upload, visualization, and MDR prediction.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.clean_data import DataCleaner
from src.features.engineer_features import FeatureEngineer

# Page configuration
st.set_page_config(
    page_title="AMR Surveillance Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models and feature engineer."""
    models_dir = Path(__file__).parent.parent.parent / "models"
    
    try:
        # Load feature engineer
        fe_path = models_dir / "feature_engineer.pkl"
        if fe_path.exists():
            feature_engineer = joblib.load(fe_path)
        else:
            st.warning(f"⚠️ Feature engineer not found at {fe_path}")
            feature_engineer = None
        
        # Load leaderboard
        leaderboard_path = models_dir / "leaderboard.csv"
        if leaderboard_path.exists():
            leaderboard = pd.read_csv(leaderboard_path, index_col=0)
        else:
            st.warning(f"⚠️ Leaderboard not found at {leaderboard_path}")
            leaderboard = None
        
        # Load best model
        best_model_name = leaderboard.index[0] if leaderboard is not None else "random_forest"
        best_model_path = models_dir / f"{best_model_name}.pkl"
        if best_model_path.exists():
            best_model = joblib.load(best_model_path)
        else:
            st.warning(f"⚠️ Best model not found at {best_model_path}")
            best_model = None
        
        return {
            "feature_engineer": feature_engineer,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "leaderboard": leaderboard
        }
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("ℹ️ Dashboard will run with limited functionality. Please ensure models are trained.")
        return {
            "feature_engineer": None,
            "best_model": None,
            "best_model_name": None,
            "leaderboard": None
        }


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🦠 AMR Surveillance ML Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Data Upload & Analysis", "MDR Prediction", "Model Performance", "About"]
    )
    
    # Load models
    models_data = load_models()
    
    if page == "Home":
        show_home_page()
    elif page == "Data Upload & Analysis":
        show_upload_page(models_data)
    elif page == "MDR Prediction":
        show_prediction_page(models_data)
    elif page == "Model Performance":
        show_performance_page(models_data)
    elif page == "About":
        show_about_page()


def show_home_page():
    """Display home page with overview."""
    st.header("Welcome to the AMR Surveillance System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**📊 Data Analysis**\n\nUpload and analyze AMR surveillance data with automated cleaning and validation.")
    
    with col2:
        st.success("**🎯 MDR Prediction**\n\nPredict multi-drug resistance using state-of-the-art machine learning models.")
    
    with col3:
        st.warning("**📈 Model Performance**\n\nView comprehensive model performance metrics and leaderboard.")
    
    st.markdown("---")
    
    st.subheader("Key Features")
    
    features = {
        "Automated Data Cleaning": "Standardized antibiotic encoding, missing value handling, and MDR classification",
        "6 ML Algorithms": "Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, SVM",
        "High Accuracy": "Best model achieves 99.4% ROC-AUC with 100% MDR recall",
        "Interactive Predictions": "Single isolate or batch predictions with confidence scores",
        "Comprehensive Analysis": "Descriptive statistics, visualizations, and reporting"
    }
    
    for feature, description in features.items():
        st.markdown(f"**{feature}**: {description}")
    
    st.markdown("---")
    st.info("ℹ️ Navigate using the sidebar to access different features of the dashboard.")


def show_upload_page(models_data):
    """Display data upload and analysis page."""
    st.header("📤 Data Upload & Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with AMR surveillance data",
        type=["csv"],
        help="Upload a CSV file containing bacterial isolate data with antibiotic susceptibility results"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
            st.info(f"📊 Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Show raw data preview
            with st.expander("View Raw Data (first 10 rows)"):
                st.dataframe(df.head(10))
            
            # Clean data button
            if st.button("🔧 Clean and Process Data"):
                with st.spinner("Processing data..."):
                    cleaner = DataCleaner()
                    df_clean = cleaner.clean(df)
                    
                    st.success(f"✅ Data cleaned successfully!")
                    st.info(f"📊 Final dataset: {len(df_clean)} rows ({len(df) - len(df_clean)} dropped)")
                    
                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Isolates", len(df_clean))
                    
                    with col2:
                        if "is_mdr" in df_clean.columns:
                            mdr_count = df_clean["is_mdr"].sum()
                            st.metric("MDR Isolates", mdr_count)
                    
                    with col3:
                        if "is_mdr" in df_clean.columns:
                            mdr_rate = (df_clean["is_mdr"].sum() / len(df_clean)) * 100
                            st.metric("MDR Rate", f"{mdr_rate:.1f}%")
                    
                    with col4:
                        if "mar_index_calculated" in df_clean.columns:
                            mean_mar = df_clean["mar_index_calculated"].mean()
                            st.metric("Mean MAR Index", f"{mean_mar:.3f}")
                    
                    # Visualizations
                    st.subheader("📊 Data Visualizations")
                    
                    # Species distribution
                    if "bacterial_species" in df_clean.columns:
                        st.markdown("**Bacterial Species Distribution**")
                        species_counts = df_clean["bacterial_species"].value_counts()
                        fig = px.bar(
                            x=species_counts.index,
                            y=species_counts.values,
                            labels={"x": "Species", "y": "Count"},
                            title="Distribution of Bacterial Species"
                        )
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sample source distribution
                    if "sample_source" in df_clean.columns:
                        st.markdown("**Sample Source Distribution**")
                        source_counts = df_clean["sample_source"].value_counts()
                        fig = px.pie(
                            values=source_counts.values,
                            names=source_counts.index,
                            title="Distribution by Sample Source"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # MDR by source
                    if "is_mdr" in df_clean.columns and "sample_source" in df_clean.columns:
                        st.markdown("**MDR Rate by Sample Source**")
                        mdr_by_source = df_clean.groupby("sample_source")["is_mdr"].agg(["sum", "count"])
                        mdr_by_source["rate"] = (mdr_by_source["sum"] / mdr_by_source["count"]) * 100
                        
                        fig = px.bar(
                            x=mdr_by_source.index,
                            y=mdr_by_source["rate"],
                            labels={"x": "Sample Source", "y": "MDR Rate (%)"},
                            title="MDR Rate by Sample Source"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download cleaned data
                    st.subheader("💾 Download Processed Data")
                    csv = df_clean.to_csv(index=False)
                    st.download_button(
                        label="Download Cleaned CSV",
                        data=csv,
                        file_name="cleaned_amr_data.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


def show_prediction_page(models_data):
    """Display MDR prediction page."""
    st.header("🎯 MDR Prediction")
    
    if not models_data.get("best_model"):
        st.error("❌ Models not loaded. Please ensure models are available in the models/ directory.")
        return
    
    st.info(f"📊 Using model: **{models_data['best_model_name']}**")
    
    st.subheader("Enter Isolate Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        species = st.selectbox(
            "Bacterial Species",
            ["escherichia_coli", "klebsiella_pneumoniae_ssp_pneumoniae", 
             "enterobacter_cloacae_complex", "enterobacter_aerogenes"]
        )
        
        source = st.selectbox(
            "Sample Source",
            ["drinking_water", "river_water", "lake_water", "fish_tilapia", 
             "fish_banak", "fish_gusaw", "fish_kaolang", 
             "effluent_water_treated", "effluent_water_untreated"]
        )
    
    with col2:
        region = st.selectbox(
            "Administrative Region",
            ["region_iii_central_luzon", "region_viii_eastern_visayas", "barmm"]
        )
    
    st.subheader("Antibiotic Susceptibility Results")
    st.markdown("*Enter S (Susceptible), I (Intermediate), or R (Resistant) for each antibiotic*")
    
    # Create antibiotic input fields in columns
    antibiotics = [
        "ampicillin", "amoxicillin_clavulanic_acid", "cefalotin", 
        "gentamicin", "enrofloxacin", "tetracycline", "chloramphenicol"
    ]
    
    results = {}
    cols = st.columns(3)
    for idx, antibiotic in enumerate(antibiotics):
        with cols[idx % 3]:
            results[f"{antibiotic}_int"] = st.selectbox(
                f"{antibiotic.replace('_', ' ').title()}",
                ["s", "i", "r", ""],
                key=antibiotic
            )
    
    if st.button("🔮 Predict MDR", type="primary"):
        try:
            # Create input dataframe
            input_data = {
                "bacterial_species": species,
                "sample_source": source,
                "administrative_region": region,
                **results
            }
            
            df_input = pd.DataFrame([input_data])
            
            # Clean and process
            cleaner = DataCleaner()
            df_clean = cleaner.clean(df_input)
            
            # Engineer features
            fe = models_data["feature_engineer"]
            X, _, _ = fe.prepare_for_modeling(df_clean, target_col=None, fit=False, scale=True)
            
            # Predict
            model = models_data["best_model"]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[0, 1]
            else:
                prob = model.predict(X)[0]
            
            classification = "MDR" if prob >= 0.5 else "Non-MDR"
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Classification", classification)
            
            with col2:
                st.metric("MDR Probability", f"{prob:.1%}")
            
            with col3:
                confidence = "High" if prob >= 0.8 or prob <= 0.2 else "Medium" if prob >= 0.6 or prob <= 0.4 else "Low"
                st.metric("Confidence", confidence)
            
            # Visual gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "MDR Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if prob >= 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            if classification == "MDR":
                st.error("⚠️ **High Risk**: This isolate is classified as Multi-Drug Resistant")
            else:
                st.success("✅ **Low Risk**: This isolate is not classified as MDR")
        
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")


def show_performance_page(models_data):
    """Display model performance page."""
    st.header("📈 Model Performance")
    
    if models_data.get("leaderboard") is not None:
        st.subheader("🏆 Model Leaderboard")
        
        leaderboard = models_data["leaderboard"]
        
        # Display key metrics
        st.dataframe(
            leaderboard[["accuracy", "balanced_accuracy", "recall", "f1", "roc_auc", "composite_score"]].style.format("{:.4f}"),
            use_container_width=True
        )
        
        st.info(f"🥇 Best Model: **{models_data['best_model_name']}** (Composite Score: {leaderboard.loc[models_data['best_model_name'], 'composite_score']:.4f})")
        
        # Visualize model comparison
        st.subheader("Model Comparison")
        
        metrics_to_plot = ["roc_auc", "recall", "f1", "balanced_accuracy"]
        
        fig = go.Figure()
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric.upper().replace("_", " "),
                x=leaderboard.index,
                y=leaderboard[metric]
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ Leaderboard data not available")


def show_about_page():
    """Display about page."""
    st.header("ℹ️ About the AMR Surveillance System")
    
    st.markdown("""
    ### Project Overview
    
    This system provides a comprehensive solution for Antimicrobial Resistance (AMR) surveillance 
    using machine learning to predict Multi-Drug Resistant (MDR) bacterial isolates from 
    water-fish-human surveillance data.
    
    ### Key Components
    
    1. **Data Processing Pipeline**
       - Automated cleaning and validation
       - Standardized antibiotic encoding (S→0, I→1, R→2)
       - MDR classification (≥3 resistant antibiotic classes)
       - MAR index calculation
    
    2. **Machine Learning Models**
       - 6 algorithms trained and compared
       - Best model: Random Forest
       - Excellent performance: 99.4% ROC-AUC
       - 100% recall for MDR detection
    
    3. **REST API**
       - FastAPI-based service
       - Endpoints for prediction, batch processing, and data upload
       - Swagger documentation at `/docs`
    
    4. **Interactive Dashboard**
       - This Streamlit application
       - Data visualization and analysis
       - Real-time MDR prediction
    
    ### Model Performance
    
    The best performing model (Random Forest) achieves:
    - **ROC-AUC**: 99.4%
    - **Recall**: 100% (detects all MDR cases)
    - **F1-Score**: 69.6%
    - **Balanced Accuracy**: 96.1%
    
    ### MDR Definition
    
    An isolate is classified as **Multi-Drug Resistant (MDR)** if it shows resistance 
    to antibiotics from **3 or more** different antibiotic classes.
    
    ### Technology Stack
    
    - **ML Framework**: scikit-learn, XGBoost, LightGBM
    - **API**: FastAPI
    - **Dashboard**: Streamlit
    - **Deployment**: Docker, Docker Compose
    - **CI/CD**: GitHub Actions
    
    ### Version Information
    
    - **Version**: 0.1.0
    - **Last Updated**: December 2025
    - **License**: MIT
    
    ### Contact & Support
    
    For questions or support, please refer to the project repository.
    """)


if __name__ == "__main__":
    main()
