import streamlit as st

def page_summary_body():
    st.title("🩸 Thalassemia Predictor")
    st.markdown("---")
    
    st.header("Project Overview")
    st.write("""
    This machine learning application predicts thalassemia carrier status using Complete Blood Count (CBC) parameters 
    and hemoglobin electrophoresis results. The system is designed to assist healthcare professionals in early 
    screening and diagnosis of thalassemia carriers.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Business Requirements")
        st.write("""
        **BR1:** Understand CBC patterns that distinguish thalassemia carriers from normal individuals
        
        **BR2:** Develop a predictive model for thalassemia screening using clinical parameters
        
        **BR3:** Create an interactive dashboard for clinical decision support
        """)
        
        st.subheader("🔬 Clinical Context")
        st.write("""
        **Thalassemia** is a genetic blood disorder affecting hemoglobin production. Early detection is crucial for:
        - Genetic counseling
        - Family planning decisions
        - Preventing severe forms in offspring
        - Appropriate medical management
        """)
    
    with col2:
        st.subheader("📊 Dataset Information")
        st.write("""
        - **Total Samples:** 203 patients
        - **Features:** 16 clinical parameters
        - **Target:** Thalassemia carrier status
        - **Key Parameters:** HbA, HbA2, HbF, MCV, MCH, RBC indices
        """)
        
        st.subheader("🤖 ML Approach")
        st.write("""
        - **Feature Engineering:** Clinical indicators (Mentzer Index, microcytosis, hypochromia)
        - **Models:** XGBoost, Random Forest, Gradient Boosting, Logistic Regression
        - **Validation:** 5-fold stratified cross-validation
        - **Best Model:** XGBoost (F1-score: 0.685)
        """)
    
    st.markdown("---")
    st.subheader("🔍 Key Clinical Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mentzer Index", "< 13", help="MCV/RBC ratio - Gold standard for thalassemia screening")
        st.metric("Microcytosis", "MCV < 80 fL", help="Small red blood cell size")
    
    with col2:
        st.metric("Hypochromia", "MCH < 27 pg", help="Reduced hemoglobin content per cell")
        st.metric("HbA2 Levels", "1.5-3.5%", help="Elevated in beta-thalassemia trait")
    
    with col3:
        st.metric("HbF Levels", "< 2%", help="Fetal hemoglobin persistence")
        st.metric("RBC/Hb Ratio", "Variable", help="Red cell count to hemoglobin ratio")
    
    st.markdown("---")
    st.info("""
    **Disclaimer:** This tool is for educational and research purposes only. 
    It should not replace professional medical diagnosis or clinical judgment.
    """)
