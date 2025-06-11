# src/data_validator.py
import pandas as pd
import tensorflow_data_validation as tfdv

def validate_data(dataframe_to_validate: pd.DataFrame, schema, training_stats) -> bool:
    """
    Valide un DataFrame par rapport à un schéma et des statistiques de référence.
    Détecte les anomalies de type et de distribution (skew).
    """
    print(f"\n--- Validation d'un jeu de données (taille: {len(dataframe_to_validate)}) ---")
    
    serving_stats = tfdv.generate_statistics_from_dataframe(dataframe_to_validate)
    
    anomalies = tfdv.validate_statistics(
        statistics=training_stats,
        schema=schema, 
        serving_statistics=serving_stats
    )
    
    if anomalies.anomaly_info:
        print("\n>>> VALIDATION ÉCHOUÉE : Des anomalies ont été détectées.")
        tfdv.display_anomalies(anomalies)
        return False
    else:
        print("\n>>> VALIDATION RÉUSSIE : Aucune anomalie détectée.")
        return True