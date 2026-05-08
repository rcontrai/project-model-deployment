import streamlit as st
import httpx
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

# Configuration générale
API_URL = "http://localhost:8000"
ASSETS_DIR = "./ui_assets/"

# Fonctions d'appel à l'API
@st.cache_data
def get_application_id_limits():
    with httpx.Client() as client:
        response = client.get(API_URL + "/get_application_id_limits")
        response.raise_for_status()
        return response

@st.cache_data
def get_decision_threshold():
    with httpx.Client() as client:
        response = client.get(API_URL + "/get_decision_threshold")
        response.raise_for_status()
        return response

@st.cache_data
def get_application_data(sk_id_curr:int):
    with httpx.Client() as client:
        response = client.post(API_URL + "/get_application_data",
                                json={"sk_id_curr":str(sk_id_curr)})
        response.raise_for_status()
        return response

@st.cache_data
def get_prediction(predict_body:dict):
    with httpx.Client() as client:
        response = client.post(API_URL + "/predict", json=predict_body)
        response.raise_for_status()
        return response

# Fonctions de chargement de ressources
@st.cache_data
def load_percentiles()->tuple[np.ndarray, np.ndarray]:
    data = np.load(os.path.join(ASSETS_DIR, "percentiles.npz"))
    percentiles_neg = data["percentiles_neg"]
    percentiles_pos = data["percentiles_pos"]
    return percentiles_neg, percentiles_pos

@st.cache_data
def load_figure(filename:str):
    with open(os.path.join(ASSETS_DIR, filename), 'rb') as f:
        figure = pickle.load(f)
    return figure

# Paramètres des entrées
# Copié-collé des dictionnaires définis dans src.feature_engineering_small.shrink_app
# On pourrait les rendre partagés si on était prêt à partager du code entre le frontend et le backend
# Ou alors faire un endpoint d'API qui les transmet
name_contract_type_dict = {"Cash loans":False, "Revolving loans":True}
code_gender_dict = {"F":0, "M":1, "XNA":2}
flag_own_car_dict = {"N":False, "Y":True}
name_eduction_type_dict = {"Lower secondary" : 0,
                        "Secondary / secondary special" : 1,
                        "Incomplete higher" : 2,
                        "Higher education" : 3,
                        "Academic degree" : 4}
name_family_status_dict = {'Married': 0,
                        'Single / not married': 1,
                        'Civil marriage': 2,
                        'Separated': 3,
                        'Widow': 4,
                        'Unknown': 5}
name_contract_type_options = tuple(name_contract_type_dict.keys())
code_gender_options = tuple(code_gender_dict.keys())
flag_own_car_options = tuple(flag_own_car_dict.keys())
name_eduction_type_options = tuple(name_eduction_type_dict.keys())
name_family_status_options = tuple(name_family_status_dict.keys())

sk_id_curr_limits = get_application_id_limits().json()

# Fonctions internes
def sanitize_int(value, name:str):
    if value is None:
        raise ValueError(f"{name} cannot be empty")
    try:
        return int(value)
    except TypeError as e:
        raise(f"Could not convert value {value} of {name} into an integer:\n\"{e}\"")

def sanitize_float(value, name:str):
    if value is None:
        raise ValueError(f"{name} cannot be empty")
    try:
        return float(value)
    except TypeError as e:
        raise ValueError(f"Could not convert value {value} of {name} into a float:\n\"{e}\"")

def sanitize_optionnal_float(value, name:str):
    if value is None:
        return None
    try:
        return float(value)
    except TypeError as e:
        raise ValueError(f"Could not convert value {value} of {name} into a float:\n\"{e}\"")

def generate_percentile_text(score:float, percentiles:np.ndarray, group_name:str, comparative:str)->str:
    if comparative == "lower":
        n_quantiles = np.sum(score >= percentiles)
    elif comparative == "higher":
        n_quantiles = np.sum(score <= percentiles)
    if n_quantiles == 0:
        proportion_block = "0%"
    elif n_quantiles == 1:
        proportion_block = "Under 1%"
    else:
        proportion = (n_quantiles - 1) / 100
        proportion_block = f"{proportion:.0%}"
    # return f"{proportion_block} of {group_name} get a risk score {comparative} {score:.1%}"
    return f"{proportion_block} of {group_name} get a {comparative} risk score"

# Contenu de la page

st.title("Default risk prediction app")

if "id_submitted" not in st.session_state:
    st.session_state.id_submitted = False
if "data_submited" not in st.session_state:
    st.session_state.data_submited = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None

tabs = st.tabs(("📝Inputs", "🧮Prediction"), key="tabs") #📊monitoring

# Formulaires d'entrée
with tabs[0]:
    with st.form("Application ID"):
        sk_id_curr = st.number_input("Application ID", value=None, 
                                    min_value=sk_id_curr_limits["min"], max_value=sk_id_curr_limits["max"],
                                    format="%.0d", step=1)
        if (sk_id_curr is not None):
            sk_id_curr = int(sk_id_curr)
        id_submitted = st.form_submit_button("Get application data")
        id_submitted = id_submitted and (sk_id_curr is not None)
        st.session_state.id_submitted = st.session_state.id_submitted or id_submitted

    if st.session_state.id_submitted:
        get_app_data_response = get_application_data(sk_id_curr)
        features_dict = get_app_data_response.json()
        with st.form("Application data"):
            data_submited = st.form_submit_button("Predict Default Risk")
            col1, col2 = st.columns(2)
            with col1:
                name_contract_type = st.selectbox("Contract Type", name_contract_type_options, index=int(features_dict["NAME_CONTRACT_TYPE"]))
                amt_credit = st.number_input("Loan Credit Amount", value=features_dict["AMT_CREDIT"], min_value=0., format="%.1d")
                amt_annuity = st.number_input("Loan Annuity", value=features_dict["AMT_ANNUITY"], min_value=0., format="%.1d")
                code_gender = st.selectbox("Gender", code_gender_options, index=features_dict["CODE_GENDER"])
                name_eduction_type = st.selectbox("Level of Education", name_eduction_type_options, index=features_dict["NAME_EDUCATION_TYPE"])
                amt_income_total = st.number_input("Income", value=features_dict["AMT_INCOME_TOTAL"], min_value=0., format="%.1d")
                days_employed = st.number_input("Duration of Current Job (in days)", value=features_dict["DAYS_EMPLOYED"], format="%.0d", step=365)
                days_birth = st.number_input("Age (in days)", value=features_dict["DAYS_BIRTH"], format="%.0d", min_value=1, step=365)
            with col2:
                ext_source_1 = st.number_input("Credit score 1", value=features_dict["EXT_SOURCE_1"], min_value=0., max_value=1., format="%.4f")
                ext_source_2 = st.number_input("Credit score 2", value=features_dict["EXT_SOURCE_2"], min_value=0., max_value=1., format="%.4f")
                ext_source_3 = st.number_input("Credit score 3", value=features_dict["EXT_SOURCE_3"], min_value=0., max_value=1., format="%.4f")
                name_family_status = st.selectbox("Client's Family Status", name_family_status_options, index=features_dict["NAME_FAMILY_STATUS"])
                st.space("small")
                flag_own_car = st.checkbox("Client Owns a Car", features_dict["FLAG_OWN_CAR"])
                st.space("xxsmall")
                # Bloc vide - haut comme un bouton + label
                st.space("xsmall")
                st.space("medium")
                days_id_publish = st.number_input("Age of Identity Document (in days)", value=features_dict["DAYS_ID_PUBLISH"], min_value=0, format="%.0d", step=365)
                days_last_phone_change = st.number_input("Age of client's Phone (in days)", value=features_dict["DAYS_LAST_PHONE_CHANGE"], min_value=0, format="%.0d", step=365)

        if data_submited:
            data_ok=False
            try :
                sk_id_curr_safe = sanitize_int(sk_id_curr, "Application ID")
                predict_body = {
                    "SK_ID_CURR": sk_id_curr_safe,
                    "NAME_CONTRACT_TYPE": name_contract_type_dict[name_contract_type],
                    "CODE_GENDER": code_gender_dict[code_gender],
                    "FLAG_OWN_CAR": flag_own_car,
                    "AMT_INCOME_TOTAL": sanitize_float(amt_income_total, "Income"),
                    "AMT_CREDIT": sanitize_float(amt_credit, "Loan Credit Amount"),
                    "AMT_ANNUITY": sanitize_optionnal_float(amt_annuity, "Loan Annuity"),
                    "NAME_EDUCATION_TYPE": name_eduction_type_dict[name_eduction_type],
                    "NAME_FAMILY_STATUS": name_family_status_dict[name_family_status],
                    "DAYS_BIRTH": sanitize_int(days_birth, "Age (in days)"),
                    "DAYS_EMPLOYED": sanitize_int(days_employed, "Duration of Current Job (in days)"),
                    "DAYS_ID_PUBLISH": sanitize_int(days_id_publish, "Age of Identity Document (in days)"),
                    "EXT_SOURCE_1": sanitize_optionnal_float(ext_source_1, "Credit score 1"),
                    "EXT_SOURCE_2": sanitize_optionnal_float(ext_source_2, "Credit score 2"),
                    "EXT_SOURCE_3": sanitize_optionnal_float(ext_source_3, "Credit score 3"),
                    "DAYS_LAST_PHONE_CHANGE": sanitize_int(days_last_phone_change, "Age of client's Phone (in days)")
                }
                data_ok = True
            except ValueError as e:
                st.error(str(e))
            if data_ok:
                predict_response = get_prediction(predict_body)
                prediction = predict_response.json()
                prediction["sk_id_curr"] = sk_id_curr_safe
                st.session_state.prediction = prediction

# Résultats
with tabs[1]:
    if st.session_state.prediction is None:
        st.markdown("*no prediction yet*")
    else:
        prediction = st.session_state.prediction
        decison_text = "❌Reject" if prediction["prediction"] else "✅Accept"
        threshold = get_decision_threshold().json()["threshold"]
        score = prediction["probability"]
        st.markdown(f"*Prediction for application \\#{prediction["sk_id_curr"]}*")
        st.markdown("**Decision**: " + decison_text)
        st.markdown(f"**Risk score**: {score:.1%}" +
                    f"\\\n:small[*decision threshold: {threshold:.1%}*]")
        percentiles_neg, percentiles_pos = load_percentiles()
        col1, col2 = st.columns(2)
        with col1:
            fig = load_figure("figure_noissue_distrib.pickle")
            plt.figure(fig)
            plt.xlabel("Risk score")
            plt.title("Distribution of risk scores for clients with no issues")
            plt.axvline(threshold, linestyle="--", c="gray", zorder=0, label="threshold")
            plt.axvline(score, linestyle="--", c="blue", zorder=2, label="predicted score")
            plt.legend()
            fig = plt.gcf()
            st.pyplot(fig)
            percentile_text = generate_percentile_text(score, percentiles_neg, "clients with no issues", "higher")
            st.markdown(percentile_text)
        with col2:
            fig = load_figure("figure_default_distrib.pickle")
            plt.figure(fig)
            plt.xlabel("Risk score")
            plt.title("Distribution of risk scores for clients who default")
            plt.axvline(threshold, linestyle="--", c="gray", zorder=0, label="threshold")
            plt.axvline(score, linestyle="--", c="blue", zorder=2, label="predicted score")
            plt.legend()
            fig = plt.gcf()
            st.pyplot(fig, clear_figure=True)
            percentile_text = generate_percentile_text(score, percentiles_pos, "clients who default", "lower")
            st.markdown(percentile_text)
