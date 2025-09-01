"""
Enhanced ML utilities for thalassemia prediction
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

class ThalassemiaPredictor:
    """Enhanced thalassemia prediction class with all ML improvements"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.optimal_threshold = 0.5
        self.feature_names = None
        
    def load_model(self, model_path):
        """Load enhanced model components"""
        try:
            self.model = joblib.load(f"{model_path}/best_xgb_model.pkl")
            self.scaler = joblib.load(f"{model_path}/feature_scaler.pkl")
            self.label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")
            
            with open(f"{model_path}/model_config.json", 'r') as f:
                config = json.load(f)
                self.optimal_threshold = config.get('optimal_threshold', 0.5)
                self.feature_names = config.get('features', [])
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess input data with feature engineering and scaling"""
        df = pd.DataFrame([input_data])
        
        # Apply feature engineering
        df = self._engineer_features(df)
        
        # Select only model features
        if self.feature_names:
            df = df[self.feature_names]
        
        # Apply scaling
        if self.scaler:
            numerical_features = ['hba2', 'hbf', 'mcv', 'mentzer_index', 'rbc_hb_ratio']
            available_numerical = [f for f in numerical_features if f in df.columns]
            if available_numerical:
                df[available_numerical] = self.scaler.transform(df[available_numerical])
        
        return df
    
    def _engineer_features(self, df):
        """Apply clinical feature engineering"""
        # Basic clinical indicators
        if 'mcv' in df.columns and 'rbc' in df.columns:
            df['mentzer_index'] = df['mcv'] / df['rbc']
        if 'mcv' in df.columns:
            df['microcytosis'] = (df['mcv'] < 80).astype(int)
        if 'mch' in df.columns:
            df['hypochromia'] = (df['mch'] < 27).astype(int)
        if 'hba2' in df.columns:
            df['hba2_elevated'] = (df['hba2'] > 3.5).astype(int)
        if 'rbc' in df.columns and 'hb' in df.columns:
            df['rbc_hb_ratio'] = df['rbc'] / df['hb']
        
        return df
    
    def predict(self, input_data):
        """Make prediction with optimal threshold"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess input
        processed_data = self.preprocess_input(input_data)
        
        # Get prediction probabilities
        proba = self.model.predict_proba(processed_data)[0]
        
        # Apply optimal threshold
        prediction = (proba[1] >= self.optimal_threshold).astype(int)
        
        # Convert to string label
        if self.label_encoder:
            prediction_label = self.label_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = 'alpha carrier' if prediction == 1 else 'normal'
        
        return {
            'prediction': prediction_label,
            'probability': proba,
            'confidence': max(proba) * 100,
            'threshold_used': self.optimal_threshold
        }

def evaluate_model_performance(model, X_test, y_test, label_encoder=None):
    """Comprehensive model evaluation"""
    predictions = model.predict(X_test)
    
    if label_encoder:
        y_test_labels = label_encoder.inverse_transform(y_test)
        pred_labels = label_encoder.inverse_transform(predictions)
    else:
        y_test_labels = y_test
        pred_labels = predictions
    
    # Calculate metrics
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    f1_macro = f1_score(y_test, predictions, average='macro')
    
    # Classification report
    report = classification_report(y_test_labels, pred_labels, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    return {
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': report,
        'confusion_matrix': cm
    }

def plot_feature_importance(feature_names, importance_scores, save_path=None):
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importance_scores)[::-1]
    
    plt.bar(range(len(importance_scores)), importance_scores[indices])
    plt.xticks(range(len(importance_scores)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def optimize_threshold(model, X_val, y_val, metric='f1'):
    """Optimize prediction threshold for clinical use"""
    from sklearn.metrics import precision_recall_curve
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    
    if metric == 'f1':
        # Calculate F1 scores
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_score = f1_scores[optimal_idx]
    else:
        # Default to 0.5
        optimal_threshold = 0.5
        optimal_score = None
    
    return optimal_threshold, optimal_score

def create_clinical_report(prediction_result, patient_data):
    """Generate clinical interpretation report"""
    report = {
        'prediction': prediction_result['prediction'],
        'confidence': prediction_result['confidence'],
        'clinical_indicators': {},
        'recommendations': []
    }
    
    # Clinical indicators
    if 'mcv' in patient_data and 'rbc' in patient_data:
        mentzer_index = patient_data['mcv'] / patient_data['rbc']
        report['clinical_indicators']['mentzer_index'] = {
            'value': mentzer_index,
            'interpretation': 'Positive for thalassemia' if mentzer_index < 13 else 'Negative'
        }
    
    if 'mcv' in patient_data:
        report['clinical_indicators']['microcytosis'] = {
            'value': patient_data['mcv'],
            'interpretation': 'Present' if patient_data['mcv'] < 80 else 'Absent'
        }
    
    if 'mch' in patient_data:
        report['clinical_indicators']['hypochromia'] = {
            'value': patient_data['mch'],
            'interpretation': 'Present' if patient_data['mch'] < 27 else 'Absent'
        }
    
    if 'hba2' in patient_data:
        report['clinical_indicators']['hba2_elevated'] = {
            'value': patient_data['hba2'],
            'interpretation': 'Elevated' if patient_data['hba2'] > 3.5 else 'Normal'
        }
    
    if prediction_result['prediction'] == 'alpha carrier':
        report['recommendations'] = [
            'Confirm with hemoglobin electrophoresis',
            'Genetic counseling recommended',
            'Screen family members',
            'Consider iron studies to rule out iron deficiency',
            'Refer to hematologist if indicated'
        ]
    else:
        report['recommendations'] = [
            'No immediate action required for thalassemia',
            'Consider other causes if anemia is present',
            'Routine follow-up as clinically indicated'
        ]
    
    return report
