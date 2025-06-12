# src/data_processing.py
import pandas as pd
import numpy as np
import joblib
import os
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import config

class DataPreprocessor:
    """
    Classe pour charger, prétraiter et préparer les données sur l'obésité.
    """
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique les transformations et crée de nouvelles caractéristiques."""
        processed_df = df.copy()

        # Transformations de base
        processed_df['Age'] = np.ceil(processed_df['Age']).astype(int)
        processed_df['Gender'] = processed_df['Gender'].map({'Male': 0, 'Female': 1})
        processed_df['family_history_with_overweight'] = processed_df['family_history_with_overweight'].map({'yes': 1, 'no': 0})
        processed_df['FAVC'] = processed_df['FAVC'].map({'yes': 1, 'no': 0})
        processed_df['FCVC'] = processed_df['FCVC'].round().astype(int)
        processed_df['NCP'] = processed_df['NCP'].round().astype(int)
        processed_df['CH2O'] = processed_df['CH2O'].round().astype(int)
        processed_df['FAF'] = processed_df['FAF'].round().astype(int)
        
        # Mappings
        processed_df['transport_activity_level'] = processed_df['MTRANS'].map(config.TRANSPORT_MAP)
        processed_df['eating_between_meals_numeric'] = processed_df['CAEC'].map(config.EATING_FREQ_MAP)
        processed_df['alcohol_numeric'] = processed_df['CALC'].map(config.EATING_FREQ_MAP)
        
        # Feature Engineering
        processed_df['Age_squared'] = processed_df['Age']**2
        processed_df['Is_Young'] = (processed_df['Age'] < 30).astype(int)
        processed_df['Is_MiddleAge'] = ((processed_df['Age'] >= 30) & (processed_df['Age'] <= 50)).astype(int)
        processed_df['genetic_diet_risk'] = processed_df['family_history_with_overweight'] * processed_df['FAVC']
        processed_df['healthy_score'] = processed_df['FCVC'] + processed_df['CH2O']
        processed_df['unhealthy_score'] = processed_df['FAVC'] + processed_df['alcohol_numeric']
        processed_df['activity_calorie_balance'] = processed_df['FAF'] - processed_df['NCP']
        processed_df['sedentary_risk'] = (processed_df['TUE'].round().astype(int) > 1).astype(int)
        
        return processed_df

    def preprocess_for_training(self):
        """
        Exécute le pipeline complet de prétraitement pour l'entraînement.
        """
        print("--- Début du prétraitement pour l'entraînement ---")
        os.makedirs(config.MODELS_DIR, exist_ok=True)

        # 1. Encodage de la cible et sauvegarde de l'encodeur
        target_encoder = LabelEncoder().fit(self.df['NObeyesdad'])
        joblib.dump(target_encoder, os.path.join(config.MODELS_DIR, config.TARGET_ENCODER_FILENAME))
        
        y = target_encoder.transform(self.df['NObeyesdad'])

        # 2. Feature Engineering
        df_processed = self._apply_feature_engineering(self.df)
        
        # 3. Supprimer les colonnes inutiles et ordonner
        df_processed = df_processed.drop(columns=config.COLUMNS_TO_DROP_PRE_TRAINING, errors='ignore')
        X = df_processed[config.FINAL_MODEL_COLUMNS]
        
        # 4. Sauvegarder les colonnes et les mappings
        joblib.dump(config.FINAL_MODEL_COLUMNS, os.path.join(config.MODELS_DIR, config.MODEL_COLUMNS_FILENAME))
        mappings = {'transport_map': config.TRANSPORT_MAP, 'eating_freq_map': config.EATING_FREQ_MAP}
        joblib.dump(mappings, os.path.join(config.MODELS_DIR, config.FEATURE_MAPPINGS_FILENAME))

        # 5. Séparation des données
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Données d'entraînement et de test créées.")
        
        # 6. Créer et sauvegarder le résumé SHAP
        print("Création du résumé SHAP...")
        summary_kmeans = shap.kmeans(self.X_train, 15)
        shap_summary_df = pd.DataFrame(summary_kmeans.data, columns=self.X_train.columns)
        joblib.dump(shap_summary_df, os.path.join(config.MODELS_DIR, config.SHAP_SUMMARY_FILENAME))
        
        print("--- Prétraitement terminé. Artefacts sauvegardés. ---")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def transform_single_prediction(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme un DataFrame d'une seule ligne (venant de l'UI) pour la prédiction.
        """
        processed_df = self._apply_feature_engineering(input_data)
        
        # S'assurer que toutes les colonnes sont présentes et dans le bon ordre
        processed_df = processed_df.reindex(columns=config.FINAL_MODEL_COLUMNS, fill_value=0)
        
        return processed_df