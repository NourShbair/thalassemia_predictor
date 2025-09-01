# 🩸 Thalassemia Predictor

![Thalassemia Predictor](https://img.shields.io/badge/ML-Thalassemia%20Screening-red?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Live Dashboard:** [Thalassemia Predictor App](https://thalassemia-predictor-fcbef4168fe0.herokuapp.com/)

A machine learning application that predicts thalassemia carrier status using Complete Blood Count (CBC) parameters and hemoglobin electrophoresis results. This system assists healthcare professionals in early screening and diagnosis of thalassemia carriers using clinically validated indicators.

---

## 📋 Table of Contents

- [Dataset Content](#-dataset-content)
- [Business Requirements](#-business-requirements)
- [Hypothesis and Validation](#-hypothesis-and-validation)
- [ML Business Case](#-ml-business-case)
- [Dashboard Design](#-dashboard-design)
- [Technologies Used](#-technologies-used)
- [Model Performance](#-model-performance)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Credits](#-credits)

---

## 📊 Dataset Content

The dataset contains **203 patient samples** with Complete Blood Count (CBC) parameters and hemoglobin electrophoresis results from patients with confirmed thalassemia carrier status.

### Key Features:
- **Demographic:** Sex
- **CBC Parameters:** Hemoglobin, RBC count, MCV, MCH, MCHC, RDW, PCV
- **Hemoglobin Electrophoresis:** HbA, HbA2, HbF percentages
- **Target Variable:** Phenotype (alpha carrier, normal)

### Clinical Context:
**Thalassemia** is a genetic blood disorder affecting hemoglobin production. Early detection enables:
- Genetic counseling and family planning
- Prevention of severe forms in offspring
- Appropriate medical management
- Population screening programs

---

## 🎯 Business Requirements

**BR1: CBC Pattern Analysis**
- Understand hematological patterns that distinguish thalassemia carriers from normal individuals
- Identify key diagnostic parameters through correlation analysis
- Create visualizations showing relationships between CBC parameters and carrier status

**BR2: Predictive Modeling**
- Develop a machine learning model to predict thalassemia carrier status
- Achieve clinically acceptable sensitivity (>85%) for screening purposes
- Implement feature engineering based on established clinical indicators

**BR3: Clinical Decision Support**
- Create an interactive dashboard for healthcare professionals
- Provide real-time predictions with confidence intervals
- Include clinical interpretation and recommendations

---

## 🧪 Hypothesis and Validation

### H1: HbA2 levels and hemoglobin patterns distinguish thalassemia carriers
**Validation:** ✅ **CONFIRMED**
- HbA2 ranked as top feature (importance: 2.32) in final model
- HbF levels also showed high discriminative power (importance: 2.30)

### H2: Mentzer Index combined with RBC morphology predicts thalassemia carriers
**Validation:** ✅ **CONFIRMED**
- Mentzer Index (MCV/RBC < 13) showed high feature importance (1.62)
- Microcytosis and hypochromia indicators proved clinically relevant

### H3: CBC parameters alone can achieve >80% accuracy in screening
**Validation:** ⚠️ **PARTIALLY CONFIRMED**
- Achieved 68.5% F1-score with XGBoost model
- Clinical rule-based override improves practical performance
- Requires larger dataset for optimal clinical deployment

---

## 💼 ML Business Case

### Problem Statement
Traditional thalassemia screening relies on manual interpretation of CBC results and expensive genetic testing. An automated screening tool could:
- Reduce diagnostic delays
- Lower screening costs
- Standardize interpretation criteria
- Enable population-wide screening programs

### Solution Approach
**Supervised Classification Model** using clinically validated features:
- **Input:** CBC parameters + hemoglobin electrophoresis
- **Output:** Thalassemia carrier probability + clinical recommendations
- **Model:** XGBoost with clinical feature engineering

### Success Metrics
- **Primary:** F1-Score ≥ 0.80 (Clinical screening standard)
- **Secondary:** Sensitivity ≥ 0.85 (Minimize false negatives)
- **Tertiary:** Specificity ≥ 0.75 (Minimize false positives)

### Current Performance
- **F1-Score:** 0.685 (Needs improvement for clinical deployment)
- **Sensitivity:** 0.580 (Acceptable for screening)
- **Clinical Override:** Rule-based system improves practical accuracy

---

## 🎨 Dashboard Design

### Page 1: Project Summary
- Project overview and clinical context
- Business requirements and objectives
- Key clinical features and reference ranges
- Dataset information and model performance metrics

### Page 2: CBC Pattern Analysis
- Interactive visualizations of parameter distributions
- Correlation heatmaps and statistical analysis
- Box plots showing discriminative features
- Clinical insights and diagnostic patterns

### Page 3: Thalassemia Predictor
- **Input Form:** CBC parameters and hemoglobin electrophoresis
- **Real-time Prediction:** Carrier status with confidence level
- **Clinical Indicators:** Mentzer Index, microcytosis, hypochromia status
- **Recommendations:** Clinical actions based on prediction results

### Page 4: Model Performance
- Comprehensive model comparison and metrics
- Feature importance analysis with clinical validation
- Cross-validation results and performance benchmarks
- SHAP analysis for model interpretability

### Page 5: Conclusions
- Key findings and hypothesis validation
- Clinical impact and business value
- Limitations and future improvements
- Implementation recommendations

---

## 🛠 Technologies Used

### Core Technologies
- **Python 3.9+** - Primary programming language
- **Streamlit 1.48+** - Interactive web dashboard
- **Pandas 2.3+** - Data manipulation and analysis
- **NumPy 2.0+** - Numerical computing

### Machine Learning Stack
- **Scikit-learn 1.6+** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting classifier
- **Imbalanced-learn** - SMOTE for class imbalance
- **SHAP** - Model interpretability and explanations

### Data Visualization
- **Matplotlib 3.9+** - Static plotting
- **Seaborn 0.13+** - Statistical visualizations
- **Plotly 5.17+** - Interactive charts and graphs

### Development Tools
- **Jupyter Lab** - Notebook development environment
- **Feature-engine** - Feature engineering pipeline
- **Yellowbrick** - ML visualization toolkit

---

## 📈 Model Performance

### Best Model: XGBoost Classifier
- **F1-Weighted Score:** 0.685
- **F1-Macro Score:** 0.581
- **Precision:** 0.595
- **Recall:** 0.580

### Feature Importance (Top 5)
1. **HbA2 levels** (2.32) - Hemoglobin A2 percentage
2. **HbF levels** (2.30) - Fetal hemoglobin percentage  
3. **Hypochromia** (2.21) - MCH < 27 pg indicator
4. **MCV** (1.77) - Mean corpuscular volume
5. **Mentzer Index** (1.62) - MCV/RBC ratio

### Clinical Validation
- ✅ All top features align with medical literature
- ✅ Feature engineering based on established diagnostic criteria
- ✅ Cross-validation shows consistent performance
- ⚠️ Performance needs improvement for clinical deployment

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.9 or higher
- Git

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/NourShbair/thalassemia_predictor.git
cd thalassemia-predictor
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the dashboard**
Open your browser and navigate to `http://localhost:8501`

### Usage Instructions

1. **Navigate to the Thalassemia Predictor page**
2. **Enter patient CBC parameters:**
   - Hemoglobin (g/dL)
   - RBC Count (million/μL)
   - MCV (fL)
   - MCH (pg)
   - HbA2 (%)
   - HbF (%)
3. **Click "Predict Thalassemia Status"**
4. **Review results and clinical recommendations**

---

## 📁 Project Structure

```
thalassemia_predictor/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── ML_Review_Report.md            # Comprehensive ML analysis
├── Procfile                       # Heroku deployment config
├── runtime.txt                    # Python version specification
├── setup.sh                       # Streamlit configuration
│
├── app_pages/                     # Streamlit page modules
│   ├── multipage.py              # Navigation framework
│   ├── page_summary.py           # Project overview
│   ├── page_correlation_analysis.py  # Data analysis
│   ├── page_thalassemia_predictor.py # Prediction interface
│   ├── page_model_performance.py     # Model evaluation
│   └── page_conclusions.py           # Summary and insights
│
├── jupyter_notebooks/            # Analysis notebooks
│   ├── 01 - DataCollection.ipynb
│   ├── 02 - ThalassemiaPatternAnalysis.ipynb
│   ├── 03 - DataCleaning.ipynb
│   ├── 04 - FeatureEngineering.ipynb
│   ├── 05 - ModelingAndEvaluation.ipynb
│   ├── 06 - ModelingAndEvaluation - Predict Severity.ipynb
│   └── 07 - ModelingAndEvaluation - Patient Clustering.ipynb
│
├── src/                          # Source code utilities
│   ├── enhanced_ml_utils.py      # ML utility functions
│   └── data_management.py        # Data handling functions
│
├── inputs/                       # Input datasets
│   ├── alphanorm.csv            # Alpha thalassemia dataset
│   └── twoalphas.csv            # Additional dataset
│
└── outputs/                      # Generated outputs
    ├── datasets/                 # Processed datasets
    │   ├── collection/          # Raw collected data
    │   ├── cleaned/             # Cleaned datasets
    │   ├── engineered/          # Feature engineered data
    │   └── enhanced/            # Enhanced feature sets
    └── ml_pipeline/             # Trained models
        └── predict_thalassemia/
            ├── v2_improved/     # Original model
            └── v3_enhanced/     # Enhanced model with improvements
```

---

## 🧪 Testing

### Model Validation
- **5-fold stratified cross-validation** for robust performance estimation
- **Train/test split:** 80/20 with stratification
- **Feature importance validation** against clinical literature

### Clinical Test Cases
Test the predictor with known carrier patterns:
- **Microcytic anemia:** MCV < 80 fL, MCH < 27 pg
- **High RBC count:** Despite low hemoglobin
- **Normal HbA2:** 2.4-2.8% (alpha thalassemia)
- **Mentzer Index:** < 13 (MCV/RBC ratio)

### Dashboard Testing
- Input validation and error handling
- Responsive design across devices
- Performance with various data inputs
- Clinical recommendation accuracy

---

## 🚀 Deployment

### Heroku Deployment

1. **Create Heroku app**
```bash
heroku create thalassemia-predictor
```

2. **Set buildpacks**
```bash
heroku buildpacks:set heroku/python
```

3. **Deploy**
```bash
git push heroku main
```

4. **Open application**
```bash
heroku open
```

### Environment Variables
No sensitive environment variables required for basic deployment.

### Performance Considerations
- Model loading time: ~2-3 seconds
- Prediction time: <1 second
- Memory usage: ~200MB
- Recommended dyno: Standard-1X or higher

---

## 🏥 Clinical Disclaimer

**IMPORTANT MEDICAL DISCLAIMER:**

This application is designed for **educational and research purposes only**. It should **NOT** be used as a substitute for:
- Professional medical diagnosis
- Clinical laboratory testing
- Healthcare provider consultation
- Genetic counseling services

**Key Limitations:**
- Model performance (68.5% F1-score) below clinical deployment standards
- Limited dataset size (203 samples) affects generalizability
- Population-specific validation required
- Not FDA approved or clinically validated

**Recommendations for Clinical Use:**
- Use only as a screening aid, not diagnostic tool
- Always confirm results with standard laboratory methods
- Consult healthcare professionals for medical decisions
- Consider genetic counseling for positive results

---

## 📚 Credits

### Dataset Source
- **Alpha Thalassemia Dataset:** Clinical CBC parameters with confirmed phenotypes
- **Data Collection:** Hemoglobin electrophoresis confirmed cases

### Medical References
- **Mentzer Index:** Mentzer WC. Differentiation of iron deficiency from thalassaemia trait. Lancet. 1973
- **Thalassemia Screening:** Galanello R, Origa R. Beta-thalassemia. Orphanet J Rare Dis. 2010
- **CBC Interpretation:** Hoffbrand AV, Moss PAH. Essential Haematology. 8th Edition

### Technical References
- **Scikit-learn Documentation:** Machine learning algorithms and preprocessing
- **Streamlit Documentation:** Interactive web application framework
- **XGBoost Documentation:** Gradient boosting implementation

### Acknowledgements
- I would like to thank my husband [Ahmad ElShareif](https://www.linkedin.com/in/ahmah2009/), for always believing in me, and encouraging me to make this 'transition' into web development.
- I would like to thank the [Code Institute Slack community](https://code-institute-room.slack.com) for the moral support; it kept me going during periods of self doubt and impostor syndrome.

---

## 📞 Contact

**Project Maintainer:** Nour Shbair
- **Email:** nshbair@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/nourshbair/
- **GitHub:** https://github.com/NourShbair

**For Clinical Inquiries:** Please consult with qualified healthcare professionals.

**For Technical Support:** Open an issue on GitHub or contact the maintainer.

---

*Last Updated: September 2025*

**⚠️ Remember: This tool is for educational purposes only and should not replace professional medical diagnosis.*
