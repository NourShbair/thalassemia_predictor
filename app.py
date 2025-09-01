import streamlit as st
from app_pages.multipage import MultiPage

# Load page modules
from app_pages.page_summary import page_summary_body
from app_pages.page_correlation_analysis import page_correlation_analysis_body
from app_pages.page_thalassemia_predictor import page_thalassemia_predictor_body
from app_pages.page_model_performance import page_model_performance_body
# from app_pages.page_conclusions import page_conclusions_body

# Create MultiPage app
app = MultiPage(app_name="Thalassemia Predictor")

# Add pages
app.add_page("📋 Project Summary", page_summary_body)
app.add_page("📊 CBC Pattern Analysis", page_correlation_analysis_body)
app.add_page("🔬 Thalassemia Predictor", page_thalassemia_predictor_body)
app.add_page("📈 Model Performance", page_model_performance_body)
# app.add_page("📝 Conclusions", page_conclusions_body)

# Run the app
app.run()
