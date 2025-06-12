# src/prediction.py
import pandas as pd
import numpy as np
import joblib
import os
import shap
import streamlit as st
import config

class ObesityPredictor:
    """
    Classe pour charger les modèles et faire des prédictions sur l'obésité.
    """
    def __init__(self, models_dir: str):
        try:
            self.model = joblib.load(os.path.join(models_dir, config.MODEL_FILENAME))
            self.target_encoder = joblib.load(os.path.join(models_dir, config.TARGET_ENCODER_FILENAME))
            self.model_columns = joblib.load(os.path.join(models_dir, config.MODEL_COLUMNS_FILENAME))
            shap_summary_df = joblib.load(os.path.join(models_dir, config.SHAP_SUMMARY_FILENAME))
            
            # Initialisation de l'explainer SHAP
            masker = shap.maskers.Independent(data=shap_summary_df)
            self.explainer = shap.Explainer(self._predict_proba_for_shap, masker)
            
        except FileNotFoundError as e:
            st.error(f"Erreur de chargement d'un fichier : {e}. Veuillez exécuter le pipeline d'entraînement.")
            st.stop()
        except Exception as e:
            st.error(f"Une erreur inattendue est survenue : {e}")
            st.stop()
    
    def _predict_proba_for_shap(self, x):
        """Fonction wrapper pour la compatibilité avec SHAP Explainer."""
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=self.model_columns)
        return self.model.predict_proba(x)

    def predict(self, processed_input: pd.DataFrame):
        """Fait une prédiction et retourne la classe textuelle."""
        prediction_encoded = self.model.predict(processed_input)
        prediction_text = self.target_encoder.inverse_transform(prediction_encoded)
        return prediction_encoded[0], prediction_text[0]

    def get_shap_explanation(self, processed_input: pd.DataFrame, predicted_class_index: int):
        """Calcule et retourne les valeurs SHAP pour une prédiction."""
        shap_explanation = self.explainer(processed_input)
        
        shap_values_for_class = shap_explanation.values[0, :, predicted_class_index]
        
        feature_importance_df = pd.DataFrame({
            'Caractéristique': self.model_columns,
            'Influence (valeur SHAP)': shap_values_for_class
        })
        feature_importance_df['Influence Absolue'] = feature_importance_df['Influence (valeur SHAP)'].abs()
        
        return feature_importance_df.sort_values(by='Influence Absolue', ascending=False).head(10)