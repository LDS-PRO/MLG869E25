import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Charger tous les actifs de prétraitement et le modèle
try:
    model = joblib.load('obesity_model.joblib')
    target_encoder = joblib.load('target_label_encoder.joblib')
    age_processing_info = joblib.load('age_processing_info.joblib')
    ordinal_mappings = joblib.load('ordinal_mappings.joblib')
    winsor_limits = joblib.load('winsor_limits.joblib')
    model_columns = joblib.load('model_columns.joblib') # Liste des colonnes attendues par le modèle
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError as e:
    st.error(f"Erreur de chargement d'un fichier d'actif : {e}. Veuillez exécuter le script d'entraînement et de sauvegarde d'abord.")
    st.stop()
except Exception as e:
    st.error(f"Une erreur s'est produite lors du chargement des actifs : {e}")
    st.stop


# Récupérer les informations pour l'âge et les mappings ordinaux
age_bins = age_processing_info['bins']
age_labels = age_processing_info['labels']
caec_map = ordinal_mappings['CAEC']
calc_map = ordinal_mappings['CALC']

# Définir les options pour les selectbox
gender_options = ['Male', 'Female']
yes_no_options = ['no', 'yes']
caec_options = list(caec_map.keys())
calc_options = list(calc_map.keys())
mtrans_options = ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike']


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
        gender_input = st.selectbox("Sexe (Gender)", options=gender_options, index=0)
        height_input = st.number_input("Taille (Height, en mètres, ex: 1.75)", min_value=1.0, max_value=2.5, value=1.70, step=0.01, format="%.2f")
        weight_input = st.number_input("Poids (Weight, en kg, ex: 70.5)", min_value=30.0, max_value=200.0, value=70.0, step=0.5, format="%.1f")
        family_history_input = st.selectbox("Antécédents familiaux de surpoids (family_history_with_overweight)", options=yes_no_options, index=1)

    with col2:
        st.subheader("🍔 Habitudes Alimentaires")
        favc_input = st.selectbox("Consommation fréquente d'aliments hypercaloriques (FAVC)", options=yes_no_options, index=1)
        fcvc_input = st.number_input("Fréquence de consommation de légumes (FCVC) (1=Jamais, 2=Parfois, 3=Toujours)", min_value=1.0, max_value=3.0, value=2.0, format="%.1f")
        ncp_input = st.number_input("Nombre de repas principaux par jour (NCP)", min_value=1.0, max_value=5.0, value=3.0, format="%.1f")
        caec_input = st.selectbox("Consommation d'aliments entre les repas (CAEC)", options=caec_options, index=1) # Default 'Sometimes'
        ch2o_input = st.number_input("Consommation d'eau quotidienne (CH2O, en litres)", min_value=0.5, max_value=4.0, value=2.0, format="%.1f")
        scc_input = st.selectbox("Suivi de la consommation de calories (SCC)", options=yes_no_options, index=0)
        calc_input = st.selectbox("Consommation d'alcool (CALC)", options=calc_options, index=1) # Default 'Sometimes'

    with col3:
        st.subheader("🚴 Activité Physique et Mode de Vie")
        smoke_input = st.selectbox("Fumeur (SMOKE)", options=yes_no_options, index=0)
        # Limites pour FAF et TUE basées sur la description du dataset UCI et l'entraînement
        faf_input = st.number_input("Fréquence d'activité physique (FAF, jours/semaine)", min_value=0.0, max_value=3.0, value=1.0, format="%.1f") # Le modèle a vu 0-3
        tue_input = st.number_input("Temps d'utilisation d'appareils technologiques (TUE, heures/jour)", min_value=0.0, max_value=2.0, value=1.0, format="%.1f") # Le modèle a vu 0-2
        mtrans_input = st.selectbox("Moyen de transport principal (MTRANS)", options=mtrans_options, index=0)

    submit_button = st.form_submit_button(label="🧬 Estimer mon niveau d'obésité", use_container_width=True)


# --- Traitement après soumission ---
if submit_button:
    # 1. Créer un DataFrame à partir des entrées utilisateur
    # Les noms de colonnes doivent correspondre à ceux du DataFrame original avant traitement
    input_data_dict = {
        'Gender': [gender_input],
        'Age': [age_input], # Sera transformé
        'Height': [height_input],
        'Weight': [weight_input],
        'family_history_with_overweight': [family_history_input],
        'FAVC': [favc_input],
        'FCVC': [fcvc_input],
        'NCP': [ncp_input],
        'CAEC': [caec_input],
        'SMOKE': [smoke_input],
        'CH2O': [ch2o_input],
        'SCC': [scc_input],
        'FAF': [faf_input],
        'TUE': [tue_input],
        'CALC': [calc_input],
        'MTRANS': [mtrans_input]
    }
    input_df = pd.DataFrame.from_dict(input_data_dict)

    st.markdown("---")
    st.subheader("🧪 Traitement des données en cours...")

    # 2. Appliquer le même prétraitement que pour les données d'entraînement
    # a. Ingénierie de Caractéristiques
    input_df['BMI'] = input_df['Weight'] / (input_df['Height']**2)
    input_df['Age_Category'] = pd.cut(input_df['Age'], bins=age_bins, labels=age_labels, right=False)
    input_df.drop('Age', axis=1, inplace=True)

    # b. Encodage Binaire
    binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    for col in binary_cols:
        input_df[col] = input_df[col].map({'yes': 1, 'no': 0})

    # c. Encodage Ordinal
    input_df['CAEC'] = input_df['CAEC'].map(caec_map)
    input_df['CALC'] = input_df['CALC'].map(calc_map)

    # d. Encodage One-Hot pour nominales (Gender, MTRANS, Age_Category)
    nominal_cols_to_encode = ['Gender', 'MTRANS', 'Age_Category']
    input_df = pd.get_dummies(input_df, columns=nominal_cols_to_encode, drop_first=True)

    # e. S'assurer que toutes les colonnes du modèle sont présentes et dans le bon ordre
    # Ajouter les colonnes manquantes (qui seraient des catégories non présentes dans cette instance) avec des 0
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns] # Réorganiser/sélectionner les colonnes dans le bon ordre

    # f. Clipping (simulant le winsorizing sur une seule instance)
    for col_to_clip, limits in winsor_limits.items():
        min_val, max_val = limits
        input_df[col_to_clip] = np.clip(input_df[col_to_clip], min_val, max_val)

    # g. Mise à l'échelle des variables numériques
    # Identifier les colonnes numériques qui ont été scalées pendant l'entraînement
    # scaler.feature_names_in_ contient les noms des colonnes sur lesquelles le scaler a été ajusté
    numerical_cols_for_scaling_ui = [col for col in input_df.columns if col in scaler.feature_names_in_]

    if numerical_cols_for_scaling_ui:
        input_df[numerical_cols_for_scaling_ui] = scaler.transform(input_df[numerical_cols_for_scaling_ui])
    
    st.success("Données traitées avec succès !")

    # 3. Faire la prédiction
    try:
        prediction_encoded = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df) # Probabilités pour chaque classe

        # 4. Afficher le résultat
        prediction_text = target_encoder.inverse_transform(prediction_encoded)
        
        st.markdown("---")
        st.subheader(f"🎉 Résultat de l'Estimation :")
        st.info(f"Le niveau d'obésité estimé est : **{prediction_text[0]}**")

        # Afficher les probabilités (optionnel, mais intéressant)
        st.write("Probabilités par classe :")
        proba_df = pd.DataFrame(prediction_proba, columns=target_encoder.classes_)
        st.dataframe(proba_df.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.error("Vérifiez que toutes les entrées sont correctes et que le modèle et les actifs sont bien chargés.")