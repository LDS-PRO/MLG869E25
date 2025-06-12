# src/pretraitement.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import shap
from config import DATASET_PATH, MODELS_DIR

print("--- Début de la génération des artefacts de prétraitement (Avec Résumé SHAP en DataFrame) ---")

os.makedirs(MODELS_DIR, exist_ok=True)

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"ERREUR: Le fichier '{DATASET_PATH}' est introuvable.")
    exit()

df_processed = df.copy()

# --- Les étapes 1 à 5 restent identiques ---
# 1. Encodage de la Variable Cible
le_target = LabelEncoder().fit(df['NObeyesdad'].unique())
joblib.dump(le_target, os.path.join(MODELS_DIR, 'target_label_encoder.joblib'))

# 2. Mappings et transformations de base
transport_map = {'Walking': 4, 'Bike': 3, 'Public_Transportation': 2, 'Motorbike': 1, 'Automobile': 0}
eating_freq_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
mappings = {'transport_map': transport_map, 'eating_freq_map': eating_freq_map}
joblib.dump(mappings, os.path.join(MODELS_DIR, 'feature_mappings.joblib'))

# 3. Application des transformations et du feature engineering
df_processed['Age'] = np.ceil(df_processed['Age']).astype(int)
df_processed['Gender'] = df_processed['Gender'].map(lambda x: 1 if str(x).lower() == 'female' else 0)
df_processed['family_history_with_overweight'] = df_processed['family_history_with_overweight'].map(lambda x: 1 if str(x).lower() == 'yes' else 0)
df_processed['FAVC'] = df_processed['FAVC'].map(lambda x: 1 if str(x).lower() == 'yes' else 0)
df_processed['FCVC'] = df_processed['FCVC'].round().astype(int)
df_processed['NCP'] = df_processed['NCP'].round().astype(int)
df_processed['CH2O'] = df_processed['CH2O'].round().astype(int)
df_processed['FAF'] = df_processed['FAF'].round().astype(int)
df_processed['transport_activity_level'] = df_processed['MTRANS'].map(transport_map)
df_processed['eating_between_meals_numeric'] = df_processed['CAEC'].map(eating_freq_map)
df_processed['alcohol_numeric'] = df_processed['CALC'].map(eating_freq_map)
df_processed['Age_squared'] = df_processed['Age']**2
df_processed['Is_Young'] = (df_processed['Age'] < 30).astype(int)
df_processed['Is_MiddleAge'] = ((df_processed['Age'] >= 30) & (df_processed['Age'] <= 50)).astype(int)
df_processed['genetic_diet_risk'] = df_processed['family_history_with_overweight'] * df_processed['FAVC']
df_processed['healthy_score'] = df_processed['FCVC'] + df_processed['CH2O']
df_processed['unhealthy_score'] = df_processed['FAVC'] + df_processed['alcohol_numeric']
df_processed['activity_calorie_balance'] = df_processed['FAF'] - df_processed['NCP']
df_processed['sedentary_risk'] = (df_processed['TUE'].round().astype(int) > 1).astype(int)

# 4. Suppression des colonnes inutiles
columns_to_drop = ['CAEC', 'CALC', 'MTRANS', 'NObeyesdad', 'Weight', 'SMOKE', 'SCC', 'TUE']
df_processed.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# 5. Définir l'ordre exact des colonnes
final_model_columns_ordered = [
    'Gender', 'Age', 'Height', 'family_history_with_overweight', 'FAVC', 'FCVC', 
    'NCP', 'CH2O', 'FAF', 'transport_activity_level', 'eating_between_meals_numeric', 
    'alcohol_numeric', 'genetic_diet_risk', 'activity_calorie_balance', 
    'Age_squared', 'Is_Young', 'Is_MiddleAge', 'healthy_score', 'unhealthy_score', 
    'sedentary_risk'
]
df_processed = df_processed[final_model_columns_ordered]

# 6. Sauvegarde de la liste de colonnes
joblib.dump(final_model_columns_ordered, os.path.join(MODELS_DIR, 'model_columns.joblib'))


# --- ÉTAPE CORRIGÉE : Créer et sauvegarder le résumé des données pour SHAP ---
print("Création du résumé des données d'entraînement pour SHAP...")
# On utilise shap.kmeans pour créer un petit jeu de données de référence
summary_dense_data = shap.kmeans(df_processed, 15)

# --- CORRECTION ---
# Convertir l'objet DenseData en DataFrame pandas avant de sauvegarder.
# L'attribut .data contient les données sous forme de tableau numpy.
shap_summary_df = pd.DataFrame(summary_dense_data.data, columns=df_processed.columns)
joblib.dump(shap_summary_df, os.path.join(MODELS_DIR, 'shap_summary.joblib'))
print("Résumé SHAP (en tant que DataFrame) sauvegardé.")

print(f"\n--- Tous les artefacts de prétraitement ont été générés et sauvegardés dans {MODELS_DIR} ---")
