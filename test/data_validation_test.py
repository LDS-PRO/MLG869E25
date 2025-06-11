import pandas as pd
from src.config import DATASET_PATH, GENERATED_SCHEMA_PATH
from src.schema_generator import generate_and_save_schema
from src.data_validator import validate_data


"""
Orchestre la génération du schéma et la démonstration de la validation.
"""
# Étape 1 : Générer et sauvegarder le schéma. Récupérer le schéma et les stats.
schema, training_stats = generate_and_save_schema(DATASET_PATH, GENERATED_SCHEMA_PATH)
    
df_original = pd.read_csv(DATASET_PATH)

# --- Démonstrations de validation ---
    
print("\n" + "="*50)
print("DÉMONSTRATION 1 : Valider les données originales (doit réussir)")
validate_data(
    dataframe_to_validate=df_original,
    schema=schema,
    training_stats=training_stats
)
    
print("\n" + "="*50)
print("DÉMONSTRATION 2 : Valider des données avec des erreurs de type/plage (doit échouer)")
df_bad = pd.DataFrame([
    {'Gender': 'Other', 'Age': 250, 'Height': None, 'Weight': 75.0, 'NObeyesdad': 'Normal_Weight'},
# Remplir avec d'autres valeurs valides pour les autres colonnes pour un test complet
])
# Note : Pour un test complet, df_bad devrait avoir toutes les colonnes attendues.
# Cette version simplifiée échouera car des colonnes sont manquantes.
validate_data(
    dataframe_to_validate=df_bad,
    schema=schema,
    training_stats=training_stats
)

print("\n" + "="*50)
print("DÉMONSTRATION 3 : Valider des données avec un biais de distribution (doit échouer)")
df_highly_skewed = df_original.copy()
df_highly_skewed['Gender'] = 'Female'
validate_data(
    dataframe_to_validate=df_highly_skewed,
    schema=schema,
    training_stats=training_stats
)
