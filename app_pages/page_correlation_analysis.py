import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def page_correlation_analysis_body():
    st.title("📊 CBC Pattern Analysis")
    st.markdown("---")
    
    st.write("""
    This page analyzes the relationships between Complete Blood Count (CBC) parameters 
    and thalassemia carrier status to identify key diagnostic patterns.
    """)
    
    # Load data
    try:
        df = pd.read_csv('outputs/datasets/collection/alphanorm.csv')
        st.success(f"✅ Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
    except FileNotFoundError:
        st.error("❌ Dataset not found. Please ensure the data file exists in outputs/datasets/collection/")
        return
    
    # Data overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        carriers = df[df['phenotype'] == 'alpha carrier'].shape[0]
        st.metric("Carriers", carriers)
    with col4:
        normal = df[df['phenotype'] == 'normal'].shape[0]
        st.metric("Normal", normal)
    
    # Feature distributions
    st.subheader("🔍 Key Parameter Distributions")
    
    # Select parameters to analyze
    key_params = ['hb', 'mcv', 'mch', 'rdw', 'hba2', 'hbf']
    available_params = [param for param in key_params if param in df.columns]
    
    if len(available_params) >= 4:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=available_params[:4],
            vertical_spacing=0.1
        )
        
        colors = {'alpha carrier': '#FF6B6B', 'normal': '#4ECDC4'}
        
        for i, param in enumerate(available_params[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            for phenotype in df['phenotype'].unique():
                data = df[df['phenotype'] == phenotype][param]
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name=f"{phenotype}",
                        opacity=0.7,
                        marker_color=colors.get(phenotype, '#95A5A6'),
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=600,
            title_text="Distribution of Key CBC Parameters by Phenotype",
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Parameter Correlations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'phenotype' in df.columns:
        # Encode phenotype for correlation
        df_corr = df.copy()
        df_corr['phenotype_encoded'] = df_corr['phenotype'].map({'normal': 0, 'alpha carrier': 1})
        numeric_cols.append('phenotype_encoded')
    
    corr_matrix = df_corr[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        cmap='RdBu_r', 
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={"shrink": .8}
    )
    plt.title('CBC Parameters Correlation Matrix')
    st.pyplot(fig)
    
    # Box plots for key discriminative features
    st.subheader("📦 Key Discriminative Features")
    
    discriminative_features = ['mcv', 'mch', 'hba2']
    available_discriminative = [f for f in discriminative_features if f in df.columns]
    
    if available_discriminative:
        cols = st.columns(len(available_discriminative))
        
        for i, feature in enumerate(available_discriminative):
            with cols[i]:
                fig = px.box(
                    df, 
                    x='phenotype', 
                    y=feature,
                    color='phenotype',
                    title=f'{feature.upper()} by Phenotype',
                    color_discrete_map=colors
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    
    summary_stats = df.groupby('phenotype')[available_params].agg(['mean', 'std', 'median']).round(2)
    st.dataframe(summary_stats, use_container_width=True)
    
    # Clinical insights
    st.subheader("🏥 Clinical Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Key Findings:**")
        st.write("""
        - **MCV (Mean Corpuscular Volume):** Carriers typically show microcytosis (MCV < 80 fL)
        - **MCH (Mean Corpuscular Hemoglobin):** Reduced in carriers due to hypochromia
        - **HbA2:** Elevated levels (>3.5%) indicate beta-thalassemia trait
        - **RBC Count:** Often elevated in carriers despite low hemoglobin
        """)
    
    with col2:
        st.write("**Diagnostic Patterns:**")
        st.write("""
        - **Mentzer Index (MCV/RBC < 13):** Highly suggestive of thalassemia
        - **Microcytic hypochromic anemia:** Classic presentation
        - **Normal or elevated RBC count:** Distinguishes from iron deficiency
        - **Hemoglobin electrophoresis:** Definitive for classification
        """)
