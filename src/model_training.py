# src/model_training.py
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import config

class ModelTrainer:
    """
    Classe pour entraîner et évaluer plusieurs modèles de classification.
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {}
        self.results = {}

    def _define_models(self):
        """Définit les modèles à entraîner."""
        xgb = XGBClassifier(objective='multi:softmax', num_class=7, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        lgbm = LGBMClassifier(objective='multiclass', num_class=7, random_state=42)
        catboost = CatBoostClassifier(loss_function='MultiClass', classes_count=7, random_state=42, verbose=0)
        
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb), ('lgbm', lgbm), ('catboost', catboost)],
            voting='soft',
            weights=[0.35, 0.35, 0.3]
        )
        
        self.models = {
            "XGBoost": xgb,
            "LightGBM": lgbm,
            "CatBoost": catboost,
            "Ensemble": ensemble
        }

    def train_and_evaluate(self):
        """Entraîne tous les modèles et stocke leurs performances."""
        self._define_models()
        print("\n--- Début de l'entraînement des modèles ---")
        
        for name, model in self.models.items():
            print(f"Entraînement du modèle : {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            self.results[name] = {
                "Accuracy": accuracy_score(self.y_test, y_pred),
                "F1 Score (macro)": f1_score(self.y_test, y_pred, average='macro'),
                "Precision (macro)": precision_score(self.y_test, y_pred, average='macro'),
                "Recall (macro)": recall_score(self.y_test, y_pred, average='macro')
            }
            print(f"Performance de {name}: {self.results[name]}")

        print("--- Entraînement terminé ---")

    def save_best_model(self):
        """Identifie et sauvegarde le meilleur modèle basé sur le F1-score."""
        if not self.results:
            raise ValueError("Les modèles doivent être entraînés et évalués d'abord.")

        best_model_name = max(self.results, key=lambda name: self.results[name]['F1 Score (macro)'])
        best_model = self.models[best_model_name]
        
        model_path = os.path.join(config.MODELS_DIR, config.MODEL_FILENAME)
        joblib.dump(best_model, model_path)
        
        print("\n" + "="*70)
        print(f"MEILLEUR MODÈLE : {best_model_name}")
        print(f"Modèle sauvegardé dans : {model_path}")
        print("="*70)

# --- Script pour exécuter le processus complet ---
def run_training_pipeline():
    # 1. Prétraiter les données
    preprocessor = DataPreprocessor(config.DATASET_PATH)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_for_training()

    # 2. Entraîner les modèles
    trainer = ModelTrainer(X_train, y_train, X_test, y_test)
    trainer.train_and_evaluate()

    # 3. Sauvegarder le meilleur modèle
    trainer.save_best_model()

if __name__ == '__main__':
    run_training_pipeline()