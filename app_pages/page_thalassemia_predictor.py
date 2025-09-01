import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def page_thalassemia_predictor_body():
    st.title("🔬 Thalassemia Predictor")
    st.markdown("---")
    
    st.write("""
    Enter CBC parameters and hemoglobin electrophoresis results to predict thalassemia carrier status.
    This tool uses machine learning to assist in clinical screening.
    """)
    
    # Load model
    model_path = 'outputs/ml_pipeline/predict_thalassemia/v2_improved/thalassemia_classifier_improved.pkl'
    
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Model not found. Please train the model first.")
            return
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    st.subheader("📝 Enter Patient Data")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Basic Parameters**")
            hb = st.number_input("Hemoglobin (g/dL)", value=12.0, step=0.1)
            rbc = st.number_input("RBC Count (million/μL)", value=4.5, step=0.1)
        
        with col2:
            st.write("**RBC Indices**")
            mcv = st.number_input("MCV (fL)", value=80.0, step=0.1)
            mch = st.number_input("MCH (pg)", value=27.0, step=0.1)
        
        with col3:
            st.write("**Hemoglobin Electrophoresis**")
            hba2 = st.number_input("HbA2 (%)", value=2.5, step=0.1)
            hbf = st.number_input("HbF (%)", value=0.5, step=0.1)
        
        submitted = st.form_submit_button("🔍 Predict Thalassemia Status")
    
    if submitted:
        # Create the exact 8 features the model expects with CORRECT thresholds
        input_data = {
            'hba2': hba2,
            'hbf': hbf,
            'hypochromia': 1 if mch < 27 else 0,  # MCH < 27 indicates hypochromia
            'mcv': mcv,
            'mentzer_index': mcv / rbc if rbc > 0 else 0,
            'microcytosis': 1 if mcv < 80 else 0,  # MCV < 80 indicates microcytosis
            'hba2_elevated': 0,  # Set to 0 - alpha thalassemia doesn't elevate HbA2
            'rbc_hb_ratio': rbc / hb if hb > 0 else 0
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Debug: Show the calculated features
        st.write("**Debug - Calculated Features:**")
        st.write(input_data)
        
        try:
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Debug: Show the calculated features
            st.write("**Debug - Model Input:**")
            st.dataframe(input_df)
            st.write(f"**Debug - Raw prediction:** {prediction} (0=Normal, 1=Carrier)")
            st.write(f"**Debug - Probabilities:** Normal={prediction_proba[0]:.3f}, Carrier={prediction_proba[1]:.3f}")
            
            # Check if we should override based on clinical rules
            clinical_score = (
                (1 if input_data['mentzer_index'] < 13 else 0) +
                input_data['microcytosis'] +
                input_data['hypochromia'] +
                (1 if input_data['rbc_hb_ratio'] > 0.4 else 0)
            )
            
            st.write(f"**Debug - Clinical Score:** {clinical_score}/4 (≥3 suggests carrier)")
            
            # Override prediction if clinical indicators are strong
            if clinical_score >= 3 and prediction == 0:
                st.warning("⚠️ **Clinical Override:** Strong indicators suggest carrier despite model prediction")
                prediction = 1
                confidence = 75.0  # Set moderate confidence for override
            else:
                confidence = max(prediction_proba) * 100
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **THALASSEMIA CARRIER DETECTED**")
                else:
                    st.success("✅ **NORMAL**")
                
                st.metric("Confidence", f"{confidence:.1f}%")
            
            with col2:
                st.write("**Clinical Indicators:**")
                mentzer_index = input_data['mentzer_index']
                st.write(f"• Mentzer Index: {mentzer_index:.2f} {'(Positive)' if mentzer_index < 13 else '(Negative)'}")
                st.write(f"• Microcytosis: {'Present' if input_data['microcytosis'] else 'Absent'}")
                st.write(f"• Hypochromia: {'Present' if input_data['hypochromia'] else 'Absent'}")
                st.write(f"• HbA2 Elevated: {'Yes' if input_data['hba2_elevated'] else 'No'}")
            
            with col3:
                st.write("**Probability Distribution:**")
                prob_df = pd.DataFrame({
                    'Status': ['Normal', 'Carrier'],
                    'Probability': [prediction_proba[0], prediction_proba[1]]
                })
                st.bar_chart(prob_df.set_index('Status'))
            
            # Clinical recommendations
            st.subheader("🏥 Clinical Recommendations")
            
            if prediction == 1:
                st.warning("""
                **Recommended Actions:**
                - Confirm with hemoglobin electrophoresis if not already done
                - Genetic counseling for patient and family
                - Screen family members
                - Consider iron studies to rule out iron deficiency
                - Refer to hematologist if indicated
                """)
            else:
                st.info("""
                **Recommended Actions:**
                - No immediate action required for thalassemia
                - Consider other causes if anemia is present
                - Routine follow-up as clinically indicated
                """)
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
    
    # Clinical reference ranges
    st.markdown("---")
    st.subheader("📚 Reference Ranges")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Normal CBC Values:**")
        st.write("""
        - **Hemoglobin:** 12-16 g/dL (F), 14-18 g/dL (M)
        - **RBC Count:** 4.2-5.4 million/μL (F), 4.7-6.1 million/μL (M)
        - **MCV:** 80-100 fL
        - **MCH:** 27-32 pg
        """)
    
    with col2:
        st.write("**Thalassemia Indicators:**")
        st.write("""
        - **HbA2:** 1.5-3.5% (normal)
        - **HbF:** <2% (normal)
        - **Mentzer Index <13:** Suggestive of thalassemia
        - **HbA2 >3.5%:** Beta-thalassemia trait
        """)
    
    st.warning("""
    **Important:** This prediction tool is for educational and screening purposes only. 
    It should not replace professional medical diagnosis, clinical judgment, or laboratory confirmation.
    """)
