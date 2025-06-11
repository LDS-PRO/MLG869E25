# src/schema_generator.py
import pandas as pd
import tensorflow_data_validation as tfdv
from src.config import CATEGORICAL_FEATURES_WITH_DOMAINS, NUMERICAL_FEATURES_WITH_RANGES, SKEW_THRESHOLD

def generate_and_save_schema(data_path: str, output_schema_path: str):
    """
    Génère un schéma TFDV à partir des données, l'affine avec des règles
    prédéfinies et le sauvegarde.
    """
    print("--- Début de la génération du schéma TFDV ---")
    
    # Charger les données
    df = pd.read_csv(data_path)
    
    # Générer les statistiques
    print("1. Génération des statistiques...")
    stats = tfdv.generate_statistics_from_dataframe(dataframe=df)
    
    # Inférer le schéma de base
    print("2. Inférence du schéma de base...")
    schema = tfdv.infer_schema(statistics=stats)
    
    # Affiner le schéma
    print("3. Affinage du schéma avec des règles personnalisées...")
    
    # Règle : Rendre toutes les features obligatoires
    for feature_name in df.columns:
        feature = tfdv.get_feature(schema, feature_name)
        feature.presence.min_fraction = 1.0
    
    # Règle : Domaines stricts pour les features catégorielles
    # Mettre à jour la config avec les valeurs uniques de la cible
    CATEGORICAL_FEATURES_WITH_DOMAINS['NObeyesdad'] = df['NObeyesdad'].unique().tolist()
    for feature_name, domain_values in CATEGORICAL_FEATURES_WITH_DOMAINS.items():
        feature = tfdv.get_feature(schema, feature_name)
        feature.ClearField('domain')
        feature.string_domain.value.extend(domain_values)
        
    # Règle : Plages logiques pour les features numériques
    for feature_name, (min_val, max_val) in NUMERICAL_FEATURES_WITH_RANGES.items():
        feature = tfdv.get_feature(schema, feature_name)
        feature.float_domain.min = min_val
        feature.float_domain.max = max_val
        
    # Règle : Comparateurs de Biais (Skew)
    for feature_name in df.columns:
        feature = tfdv.get_feature(schema, feature_name)
        feature.skew_comparator.jensen_shannon_divergence.threshold = SKEW_THRESHOLD

    print("Affinage terminé.")

    # Sauvegarder le schéma
    print(f"4. Sauvegarde du schéma final dans '{output_schema_path}'...")
    tfdv.write_schema_text(schema, output_schema_path)
    print("Schéma sauvegardé avec succès.")
    
    return schema, stats