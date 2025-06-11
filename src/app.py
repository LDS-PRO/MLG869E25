import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import requests 
import json     
from config import MODELS_DIR

# --- CHARGEMENT DES ACTIFS DE PRÉTRAITEMENT (le modèle est maintenant sur Azure) ---
try:
    # On charge tout SAUF le modèle
    target_encoder = joblib.load(os.path.join(MODELS_DIR, 'target_label_encoder.joblib'))
    age_processing_info = joblib.load(os.path.join(MODELS_DIR, 'age_processing_info.joblib'))
    ordinal_mappings = joblib.load(os.path.join(MODELS_DIR, 'ordinal_mappings.joblib'))
    winsor_limits = joblib.load(os.path.join(MODELS_DIR, 'winsor_limits.joblib'))
    model_columns = joblib.load(os.path.join(MODELS_DIR, 'model_columns.joblib')) 
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
except FileNotFoundError as e:
    st.error(f"Erreur de chargement d'un fichier d'actif : {e}. Veuillez exécuter le script de prétraitement d'abord.")
    st.stop()

# --- CHARGEMENT DES SECRETS AZURE ---
try:
    AZURE_ENDPOINT_URI = st.secrets['azure']['endpoint_uri']
    AZURE_API_KEY = st.secrets['azure']['api_key']
except (KeyError, FileNotFoundError):
    st.error("Les secrets Azure (endpoint_uri, api_key) ne sont pas configurés dans .streamlit/secrets.toml")
    st.stop()


# Récupérer les informations pour l'âge et les mappings ordinaux
age_bins = age_processing_info['bins']
age_labels = age_processing_info['labels']
caec_map = ordinal_mappings['CAEC']
calc_map = ordinal_mappings['CALC']

# ... (Le reste du code de l'interface utilisateur reste EXACTEMENT le même) ...
# ... (st.set_page_config, st.title, st.form, etc.) ...
st.set_page_config(page_title="Estimateur de Niveau d'Obésité", layout="wide")
st.title("📝 Estimateur de Niveau d'Obésité")
st.markdown("""
Entrez vos informations personnelles et habitudes de vie pour obtenir une estimation de votre niveau d'obésité.
**Disclaimer :** Cet outil fournit une estimation et ne remplace pas un avis médical professionnel.
""")

# --- Création du Formulaire ---
with st.form(key='obesity_form'):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Informations Personnelles")
        age_input = st.number_input("Âge (années)", min_value=14, max_value=100, value=25, step=1)
        gender_input = st.selectbox("Sexe (Gender)", options=['Male', 'Female'], index=0)
        height_input = st.number_input("Taille (Height, en mètres, ex: 1.75)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, format="%.2f")
        weight_input = st.number_input("Poids (Weight, en kg, ex: 70.5)", min_value=30.0, max_value=200.0, value=70.0, step=0.5, format="%.1f")
        family_history_input = st.selectbox("Antécédents familiaux de surpoids (family_history_with_overweight)", options=['no', 'yes'], index=1)

    with col2:
        st.subheader("🍔 Habitudes Alimentaires")
        favc_input = st.selectbox("Consommation fréquente d'aliments hypercaloriques (FAVC)", options=['no', 'yes'], index=1)
        fcvc_input = st.number_input("Fréquence de consommation de légumes (FCVC) (1=Jamais, 2=Parfois, 3=Toujours)", min_value=1.0, max_value=3.0, value=2.0, format="%.1f")
        ncp_input = st.number_input("Nombre de repas principaux par jour (NCP)", min_value=1.0, max_value=5.0, value=3.0, format="%.1f")
        caec_input = st.selectbox("Consommation d'aliments entre les repas (CAEC)", options=list(caec_map.keys()), index=1) # Default 'Sometimes'
        ch2o_input = st.number_input("Consommation d'eau quotidienne (CH2O, en litres)", min_value=0.5, max_value=4.0, value=2.0, format="%.1f")
        scc_input = st.selectbox("Suivi de la consommation de calories (SCC)", options=['no', 'yes'], index=0)
        calc_input = st.selectbox("Consommation d'alcool (CALC)", options=list(calc_map.keys()), index=1) # Default 'Sometimes'

    with col3:
        st.subheader("🚴 Activité Physique et Mode de Vie")
        smoke_input = st.selectbox("Fumeur (SMOKE)", options=['no', 'yes'], index=0)
        faf_input = st.number_input("Fréquence d'activité physique (FAF, jours/semaine)", min_value=0.0, max_value=3.0, value=1.0, format="%.1f") # Le modèle a vu 0-3
        tue_input = st.number_input("Temps d'utilisation d'appareils technologiques (TUE, heures/jour)", min_value=0.0, max_value=2.0, value=1.0, format="%.1f") # Le modèle a vu 0-2
        mtrans_input = st.selectbox("Moyen de transport principal (MTRANS)", options=['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'], index=0)

    submit_button = st.form_submit_button(label="🧬 Estimer mon niveau d'obésité", use_container_width=True)

# --- BLOC DE PRÉDICTION MODIFIÉ ---
if submit_button:
    # Le prétraitement des données utilisateur reste identique
    # ... (code de création de input_df et de toutes les transformations jusqu'au scaling) ...
    input_data_dict = {
        'Gender': [gender_input], 'Age': [age_input], 'Height': [height_input], 'Weight': [weight_input],
        'family_history_with_overweight': [family_history_input], 'FAVC': [favc_input], 'FCVC': [fcvc_input],
        'NCP': [ncp_input], 'CAEC': [caec_input], 'SMOKE': [smoke_input], 'CH2O': [ch2o_input],
        'SCC': [scc_input], 'FAF': [faf_input], 'TUE': [tue_input], 'CALC': [calc_input], 'MTRANS': [mtrans_input]
    }
    input_df = pd.DataFrame.from_dict(input_data_dict)
    st.markdown("---")
    st.subheader("🧪 Traitement des données en cours...")

    input_df['BMI'] = input_df['Weight'] / (input_df['Height']**2)
    input_df['Age_Category'] = pd.cut(input_df['Age'], bins=age_bins, labels=age_labels, right=False)
    input_df.drop('Age', axis=1, inplace=True)

    binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    for col in binary_cols:
        input_df[col] = input_df[col].map({'yes': 1, 'no': 0})
    input_df['CAEC'] = input_df['CAEC'].map(caec_map)
    input_df['CALC'] = input_df['CALC'].map(calc_map)

    nominal_cols_to_encode = ['Gender', 'MTRANS', 'Age_Category']
    input_df = pd.get_dummies(input_df, columns=nominal_cols_to_encode, drop_first=True)
    
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]
    
    for col_to_clip, limits in winsor_limits.items():
        min_val, max_val = limits
        input_df[col_to_clip] = np.clip(input_df[col_to_clip], min_val, max_val)

    numerical_cols_for_scaling_ui = [col for col in input_df.columns if col in scaler.feature_names_in_]
    if numerical_cols_for_scaling_ui:
        input_df[numerical_cols_for_scaling_ui] = scaler.transform(input_df[numerical_cols_for_scaling_ui])
    
    st.success("Données traitées avec succès ! Envoi à Azure pour prédiction...")

    # 3. Préparer les données pour l'API et faire la prédiction via HTTP
    try:
        # Conversion du DataFrame en une liste de listes, puis en JSON
        # Le format doit correspondre à ce que le script `score.py` attend
        input_data_json = json.dumps({'data': input_df.to_numpy().tolist()})

        # Définir les en-têtes de la requête
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {AZURE_API_KEY}'
        }

        # Envoyer la requête POST à l'endpoint Azure
        response = requests.post(AZURE_ENDPOINT_URI, data=input_data_json, headers=headers)
        response.raise_for_status()  # Lève une exception si le statut est une erreur (4xx ou 5xx)

        # 4. Afficher le résultat
        prediction_encoded = response.json() # La réponse est déjà une liste de la prédiction, ex: [1]
        prediction_text = target_encoder.inverse_transform(prediction_encoded)
        
        st.markdown("---")
        st.subheader(f"🎉 Résultat de l'Estimation (via Azure) :")
        st.info(f"Le niveau d'obésité estimé est : **{prediction_text[0]}**")
        
        # Note: Obtenir les probabilités via une API peut nécessiter de modifier
        # le script score.py pour qu'il retourne `model.predict_proba`

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de communication avec l'API Azure : {e}")
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la prédiction via Azure : {e}")