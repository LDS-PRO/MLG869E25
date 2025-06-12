# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt  # <--- Importer Altair
import config
from data_processing import DataPreprocessor
from prediction import ObesityPredictor

# --- Initialisation ---
st.set_page_config(page_title="Estimateur de Niveau d'Obésité", layout="wide")

# Initialiser les classes
preprocessor = DataPreprocessor(data_path=config.DATASET_PATH) 
predictor = ObesityPredictor(models_dir=config.MODELS_DIR)

# --- Interface Utilisateur (UI) ---
# (Le code de l'interface utilisateur reste exactement le même)
st.title("📝 Estimateur de Niveau d'Obésité")
st.markdown("""
Entrez vos informations personnelles et habitudes de vie pour obtenir une estimation de votre niveau d'obésité.  
**Disclaimer :** Cet outil fournit une estimation et ne remplace pas un avis médical professionnel.
""")

caec_options = list(config.EATING_FREQ_MAP.keys())
calc_options = list(config.EATING_FREQ_MAP.keys())
mtrans_options = list(config.TRANSPORT_MAP.keys())
yes_no_options = ['no', 'yes']
gender_options = ['Male', 'Female']

with st.form(key='obesity_form'):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("👤 Informations Personnelles")
        age_input = st.number_input("Âge (années)", min_value=14, max_value=100, value=25, step=1)
        gender_input = st.selectbox("Sexe (Gender)", options=gender_options, index=0)
        height_input = st.number_input("Taille (Height, en mètres)", min_value=1.0, max_value=2.5, value=1.70, format="%.2f")
        weight_input = st.number_input("Poids (Weight, en kg)", min_value=30.0, max_value=200.0, value=70.0, format="%.1f")
        family_history_input = st.selectbox("Antécédents familiaux de surpoids", options=yes_no_options, index=1)
    
    with col2:
        st.subheader("🍔 Habitudes Alimentaires")
        favc_input = st.selectbox("Consommation fréquente d'aliments hypercaloriques (FAVC)", options=yes_no_options, index=1)
        fcvc_input = st.slider("Fréquence de consommation de légumes (FCVC)", 1, 3, 2)
        ncp_input = st.slider("Nombre de repas principaux par jour (NCP)", 1, 4, 3)
        caec_input = st.selectbox("Consommation d'aliments entre les repas (CAEC)", options=caec_options, index=1)
        ch2o_input = st.slider("Consommation d'eau quotidienne (CH2O)", 1, 3, 2)
        calc_input = st.selectbox("Consommation d'alcool (CALC)", options=calc_options, index=1)
    
    with col3:
        st.subheader("🚴 Activité Physique et Mode de Vie")
        smoke_input = st.selectbox("Fumeur (SMOKE)", options=yes_no_options, index=0)
        scc_input = st.selectbox("Suivi de la consommation de calories (SCC)", options=yes_no_options, index=0)
        faf_input = st.slider("Fréquence d'activité physique (FAF, jours/semaine)", 0, 3, 1)
        tue_input = st.slider("Temps d'utilisation d'appareils (TUE, 0-2h)", 0, 2, 1)
        mtrans_input = st.selectbox("Moyen de transport principal (MTRANS)", options=mtrans_options, index=0)
    
    submit_button = st.form_submit_button(label="🧬 Estimer mon niveau d'obésité", use_container_width=True)

# --- Logique de Prédiction ---
if submit_button:
    # (Le code de création du DataFrame et de transformation reste le même)
    input_data_dict = {
        'Gender': [gender_input], 'Age': [age_input], 'Height': [height_input], 'Weight': [weight_input],
        'family_history_with_overweight': [family_history_input], 'FAVC': [favc_input], 
        'FCVC': [fcvc_input], 'NCP': [ncp_input], 'CAEC': [caec_input], 'CH2O': [ch2o_input],
        'SCC': [scc_input], 'FAF': [faf_input], 'TUE': [tue_input], 'CALC': [calc_input], 
        'MTRANS': [mtrans_input], 'SMOKE': [smoke_input]
    }
    input_df = pd.DataFrame.from_dict(input_data_dict)
    
    st.markdown("---")
    st.write("⚙️ Application des transformations...")
    processed_df = preprocessor.transform_single_prediction(input_df)
    st.success("Données prêtes pour la prédiction.")

    try:
        predicted_class_index, prediction_text = predictor.predict(processed_df)
        
        st.markdown("---")
        st.subheader("🎉 Résultat de l'Estimation :")
        st.info(f"Le niveau d'obésité estimé est : **{prediction_text}**")
        
        with st.expander("🔍 Voir les facteurs d'influence de la prédiction (SHAP)"):
            top_features_df = predictor.get_shap_explanation(processed_df, predicted_class_index)
            
            st.write("""
            Ce graphique montre les caractéristiques qui ont le plus contribué au résultat. Les barres sont triées par importance.
            - Une influence **<span style='color: #FF4B4B;'>positive (rouge)</span>** a poussé la prédiction vers une classe de poids supérieure.
            - Une influence **<span style='color: #1f77b4;'>négative (bleue)</span>** a poussé la prédiction vers une classe de poids inférieure.
            """, unsafe_allow_html=True)

            # --- SECTION CORRIGÉE AVEC ALTAIR ---
            # 1. Préparer les données pour le graphique
            chart_df = top_features_df[['Caractéristique', 'Influence (valeur SHAP)']].copy()
            chart_df['Influence Type'] = chart_df['Influence (valeur SHAP)'].apply(lambda x: 'Positive' if x > 0 else 'Négative')

            # 2. Créer le graphique Altair
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('Influence (valeur SHAP):Q', title="Valeur d'influence SHAP"),
                y=alt.Y('Caractéristique:N', title="Caractéristique", sort='-x'), # Trier par valeur (la plus haute en haut)
                color=alt.Color('Influence Type:N',
                                scale=alt.Scale(
                                    domain=['Positive', 'Négative'],
                                    range=['#FF4B4B', '#1f77b4']  # Rouge pour positif, Bleu pour négatif
                                ),
                                legend=alt.Legend(title="Type d'influence"))
            ).properties(
                title="Top 10 des facteurs d'influence de la prédiction"
            )
            
            # 3. Afficher le graphique et le tableau
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(
                top_features_df[['Caractéristique', 'Influence (valeur SHAP)']].style.format({'Influence (valeur SHAP)': '{:+.4f}'})
            )
            # --- FIN DE LA SECTION CORRIGÉE ---

    except Exception as e:
        st.error(f"Erreur lors de la prédiction ou de l'explication SHAP : {e}")