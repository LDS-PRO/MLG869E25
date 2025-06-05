import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats.mstats import winsorize
import joblib 

try:
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
except FileNotFoundError:
    print("Le fichier 'ObesityDataSet_raw_and_data_sinthetic.csv' n'a pas été trouvé. Veuillez vérifier le chemin.")
    exit()

df_processed = df.copy()

# 1. Encodage de la Variable Cible (NObeyesdad)
le_target = LabelEncoder()
df_processed['NObeyesdad_Encoded'] = le_target.fit_transform(df_processed['NObeyesdad'])
# Sauvegarder l'encodeur de la cible
joblib.dump(le_target, 'target_label_encoder.joblib')
target_classes = le_target.classes_ # Garder pour référence

# 2. Ingénierie de Caractéristiques
df_processed['BMI'] = df_processed['Weight'] / (df_processed['Height']**2)

age_bins = [14, 19.99, 29.99, 39.99, 49.99, df_processed['Age'].max() + 1]
age_labels = ['Adolescent (14-19)', 'Jeune Adulte (20-29)', 'Adulte (30-39)', 'Adulte Moyen (40-49)', 'Adulte Âgé (50+)']
df_processed['Age_Category'] = pd.cut(df_processed['Age'], bins=age_bins, labels=age_labels, right=False)
df_processed.drop('Age', axis=1, inplace=True)
# Sauvegarder les bins et labels pour l'âge
age_processing_info = {'bins': age_bins, 'labels': age_labels}
joblib.dump(age_processing_info, 'age_processing_info.joblib')

# 3. Encodage des Variables Catégorielles
binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
for col in binary_cols:
    df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})

caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
df_processed['CAEC'] = df_processed['CAEC'].map(caec_mapping)
df_processed['CALC'] = df_processed['CALC'].map(calc_mapping)
# Sauvegarder les mappings ordinaux
ordinal_mappings = {'CAEC': caec_mapping, 'CALC': calc_mapping}
joblib.dump(ordinal_mappings, 'ordinal_mappings.joblib')

nominal_cols_to_encode = ['Gender', 'MTRANS', 'Age_Category']
df_processed = pd.get_dummies(df_processed, columns=nominal_cols_to_encode, drop_first=True)

# 4. Traitement des Variables Numériques
numerical_features_to_process = ['Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']

# Sauvegarder les limites de winsorizing (avant scaling)
winsor_limits = {}
for col_to_winsorize in ['Weight', 'BMI']:
    if col_to_winsorize in df_processed.columns:
        # Appliquer le winsorizing
        winsored_data = winsorize(df_processed[col_to_winsorize], limits=[0.01, 0.01])
        winsor_limits[col_to_winsorize] = (winsored_data.min(), winsored_data.max())
        df_processed[col_to_winsorize] = winsored_data # Mettre à jour la colonne dans df_processed
joblib.dump(winsor_limits, 'winsor_limits.joblib')

numerical_cols_for_scaling = []
for col in numerical_features_to_process:
    if col in df_processed.columns:
        numerical_cols_for_scaling.append(col)

X = df_processed.drop(['NObeyesdad', 'NObeyesdad_Encoded'], axis=1, errors='ignore')
y = df_processed['NObeyesdad_Encoded']

# Sauvegarder les noms des colonnes de X avant la séparation (après toutes les transformations)
# C'est l'ordre et l'ensemble des colonnes que le modèle attendra
model_columns = X.columns.tolist()
joblib.dump(model_columns, 'model_columns.joblib')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Mise à l'échelle (Scaler ajusté sur X_train UNIQUEMENT)
if numerical_cols_for_scaling:
    scaler = StandardScaler()
    X_train[numerical_cols_for_scaling] = scaler.fit_transform(X_train[numerical_cols_for_scaling])
    X_test[numerical_cols_for_scaling] = scaler.transform(X_test[numerical_cols_for_scaling])
    # Sauvegarder le scaler ajusté
    joblib.dump(scaler, 'scaler.joblib')
else:
    print("Aucune colonne numérique spécifiée pour le scaling n'a été trouvée.")
# --- (Fin des modifications pour la préparation) ---

print("\n" + "="*50 + "\n")
print("Préparation pour l'entraînement du modèle...")
print(f"Colonnes du modèle X_train: {X_train.columns.tolist()}")
print(f"Dimensions de X_train: {X_train.shape}")

# Entraînement d'un modèle RandomForestClassifier simple
print("\nEntraînement du modèle RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation rapide du modèle
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print(f"Accuracy sur l'ensemble d'entraînement: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Accuracy sur l'ensemble de test: {accuracy_score(y_test, y_pred_test):.4f}")

# Sauvegarde du modèle entraîné
joblib.dump(model, 'obesity_model.joblib')
print("\nModèle et tous les actifs de prétraitement ont été sauvegardés.")