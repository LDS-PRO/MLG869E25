# src/config.py
import os

# --- Chemins du Projet ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# --- Fichiers ---
DATASET_PATH = os.path.join(DATA_DIR, 'ObesityDataSet_raw_and_data_sinthetic.csv')
MODEL_FILENAME = 'best_obesity_model_ensemble_no_bmi.pkl'
TARGET_ENCODER_FILENAME = 'target_label_encoder.joblib'
FEATURE_MAPPINGS_FILENAME = 'feature_mappings.joblib'
MODEL_COLUMNS_FILENAME = 'model_columns.joblib'
SHAP_SUMMARY_FILENAME = 'shap_summary.joblib'

# --- Mappings ---
TRANSPORT_MAP = {'Walking': 4, 'Bike': 3, 'Public_Transportation': 2, 'Motorbike': 1, 'Automobile': 0}
EATING_FREQ_MAP = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

# --- Paramètres du Modèle ---
# Ordre final des colonnes pour l'entraînement et la prédiction
FINAL_MODEL_COLUMNS = [
    'Gender', 'Age', 'Height', 'family_history_with_overweight', 'FAVC', 'FCVC',
    'NCP', 'CH2O', 'FAF', 'transport_activity_level', 'eating_between_meals_numeric',
    'alcohol_numeric', 'genetic_diet_risk', 'activity_calorie_balance',
    'Age_squared', 'Is_Young', 'Is_MiddleAge', 'healthy_score', 'unhealthy_score',
    'sedentary_risk'
]

# Colonnes à supprimer avant l'entraînement
COLUMNS_TO_DROP_PRE_TRAINING = ['CAEC', 'CALC', 'MTRANS', 'NObeyesdad', 'Weight', 'SMOKE', 'SCC', 'TUE']