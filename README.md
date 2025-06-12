# Estimateur de Niveau d'Obésité via le Machine Learning

Ce projet est une application web développée avec Streamlit qui estime le niveau d'obésité d'un individu en se basant sur ses informations personnelles et ses habitudes de vie. Le modèle de Machine Learning utilisé est un **ensemble de modèles** (XGBoost, LightGBM, CatBoost) conçu pour offrir des prédictions robustes et précises.

L'une des caractéristiques clés de cette application est son **interprétabilité** : elle utilise les valeurs SHAP (SHapley Additive exPlanations) pour expliquer quelles caractéristiques ont le plus influencé une prédiction, rendant le résultat plus transparent et compréhensible pour l'utilisateur.

![Aperçu de l'application](https://i.imgur.com/your-screenshot-url.png)
*(Remplacez le lien ci-dessus par une capture d'écran de votre application pour un rendu plus professionnel)*

## ✨ Fonctionnalités Principales

-   **Interface Web Intuitive** : Un formulaire simple et élégant créé avec Streamlit pour une saisie facile des données.
-   **Prédiction par Ensemble de Modèles** : Combine la puissance de XGBoost, LightGBM et CatBoost via un `VotingClassifier` pour une meilleure généralisation et performance.
-   **Explication des Résultats (XAI)** : Affiche un graphique des valeurs SHAP pour chaque prédiction, permettant de comprendre les facteurs d'influence positifs (qui augmentent le niveau d'obésité) et négatifs.
-   **Structure de Code Professionnelle** : Le projet est organisé en classes Python distinctes (`DataPreprocessor`, `ModelTrainer`, `ObesityPredictor`), ce qui facilite la lecture, la maintenance et l'évolution du code.
-   **Pipeline de Machine Learning Complet** : Des scripts sont fournis pour automatiser le prétraitement des données, l'entraînement des modèles et la sauvegarde des artefacts nécessaires au déploiement.

## 🛠️ Stack Technique

-   **Langage** : Python 3.x
-   **Framework Web** : Streamlit
-   **Machine Learning** : Scikit-learn, XGBoost, LightGBM, CatBoost
-   **Interprétabilité (XAI)** : SHAP
-   **Manipulation de données** : Pandas, NumPy
-   **Visualisation de données** : Altair

## 📂 Structure du Projet

Le projet est organisé de manière modulaire pour séparer clairement les responsabilités, suivant les meilleures pratiques de développement logiciel.


.
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv  # Le jeu de données brut
├── models/
│   └── (Contiendra les modèles entraînés et les artefacts .joblib après exécution)
├── src/
│   ├── init.py
│   ├── config.py                 # Fichier de configuration central (chemins, mappings, etc.)
│   ├── data_processing.py      # Classe pour le chargement, nettoyage et feature engineering
│   ├── model_training.py       # Classe et script pour l'entraînement, l'évaluation et la sauvegarde
│   └── prediction.py           # Classe pour charger les modèles et effectuer les prédictions
├── app.py                      # Le point d'entrée de l'application Streamlit
├── requirements.txt            # La liste des dépendances Python à installer
└── README.md                   # Ce fichier


## 🚀 Guide d'Installation et d'Utilisation

Suivez ces étapes pour mettre en place et lancer le projet sur votre machine locale.

### Étape 1 : Cloner le Dépôt (si applicable)

Si votre projet est sur Git, clonez-le. Sinon, assurez-vous d'être dans le bon dossier.

bash
git clone [URL_DE_VOTRE_DEPOT]
cd [NOM_DU_DOSSIER_DU_PROJET]

### Étape 2 : Créer et Activer un Environnement Virtuel

L'utilisation d'un environnement virtuel est cruciale pour isoler les dépendances de votre projet et éviter les conflits.

Sur macOS / Linux :

python3 -m venv .venv
source .venv/bin/activate

Sur Windows :

# Pour PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Pour l'Invite de commandes (cmd)
python -m venv .venv
.\.venv\Scripts\activate

Vous saurez que l'environnement est activé car le nom (.venv) apparaîtra au début de votre ligne de commande.

### Étape 3 : Installer les Dépendances

Installez toutes les bibliothèques Python requises en une seule commande grâce au fichier requirements.txt.

pip install -r requirements.txt

### Étape 4 : Lancer le Projet

Vous avez deux flux de travail possibles :

Workflow 1 : Entraîner le Modèle et Lancer l'Application (Recommandé pour la première utilisation)

Ce processus complet part des données brutes, entraîne le modèle, sauvegarde les artefacts, puis lance l'application.

## 4.1. Exécuter le Pipeline d'Entraînement

Lancez le script model_training.py. Ce script va effectuer toutes les opérations nécessaires en coulisses :

Lire les données depuis data/.

Appliquer le prétraitement et le feature engineering.

Entraîner les modèles de classification.

Évaluer les modèles et sélectionner le meilleur.

Créer le dossier models/ et y sauvegarder tous les artefacts (.pkl, .joblib).

python src/model_training.py

Attendez la fin du processus. Vous devriez voir des messages de confirmation dans votre terminal.

## 4.2. Lancer l'Application Streamlit

Maintenant que les modèles sont prêts, lancez l'application web.

streamlit run app.py

Votre navigateur web par défaut devrait s'ouvrir automatiquement sur l'adresse http://localhost:8501.

Workflow 2 : Lancer Directement l'Application (si les modèles existent déjà)

Si vous avez déjà exécuté le pipeline d'entraînement et que le dossier models/ est rempli avec les fichiers .joblib et .pkl, vous pouvez directement lancer l'application.

streamlit run app.py

Quitter l'Environnement Virtuel

Lorsque vous avez fini de travailler sur le projet, vous pouvez désactiver l'environnement virtuel avec la commande :

deactivate
