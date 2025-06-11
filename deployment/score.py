# score.py
import json
import joblib
import numpy as np
import os

# Appelé une seule fois au démarrage du service pour charger le modèle.
def init():
    global model
    # Le chemin du modèle est trouvé à partir de la variable d'environnement AZUREML_MODEL_DIR
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'obesity_model.joblib')
    model = joblib.load(model_path)

# Appelé à chaque requête HTTP reçue par l'API.
def run(raw_data):
    try:
        # Les données arrivent sous forme de chaîne JSON
        data = json.loads(raw_data)['data']
        # Les données doivent être converties en tableau numpy pour la prédiction
        data = np.array(data)
        
        # Faire la prédiction
        result = model.predict(data)
        
        # Retourner la prédiction au format JSON
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error