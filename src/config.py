import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SCHEMA_DIR = os.path.join(PROJECT_ROOT, 'schemas')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

DATASET_PATH = os.path.join(DATA_DIR, 'ObesityDataSet_raw_and_data_sinthetic.csv')
GENERATED_SCHEMA_PATH = os.path.join(SCHEMA_DIR, 'obesity_schema.pbtxt')


# --- Paramètres de Validation ---
SKEW_THRESHOLD = 0.1

# Définition des règles pour le schéma
CATEGORICAL_FEATURES_WITH_DOMAINS = {
    'Gender': ['Male', 'Female'],
    'family_history_with_overweight': ['yes', 'no'],
    'FAVC': ['yes', 'no'],
    'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],
    'SMOKE': ['yes', 'no'],
    'SCC': ['yes', 'no'],
    'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],
    'MTRANS': ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'],
    'NObeyesdad': None  # Sera rempli dynamiquement à partir des données
}

NUMERICAL_FEATURES_WITH_RANGES = {
    'Age': (10.0, 100.0), 'Height': (1.0, 2.5), 'Weight': (30.0, 200.0),
    'FCVC': (1.0, 3.0), 'NCP': (1.0, 5.0), 'CH2O': (0.0, 4.0),
    'FAF': (0.0, 7.0), 'TUE': (0.0, 24.0)
}