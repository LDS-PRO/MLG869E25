# How to run (MAC)
## Create a virtual env
python3 -m venv .venv
source .venv/bin/activate
## Install requirements
pip install -r requirements.txt

## Start the streamlit app (if you have the .joblib files)
streamlit run app.py

## Create the .joblib
python3 pretraitement.py
streamlit run app.py

## Note
This version uses a basic RandomForestClassifier, when you want to test your own model, replace the obesity_model.joblib by your own.