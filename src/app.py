# Pour l'API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
# Pour le modèle
from sklearn.pipeline import Pipeline
from data_caching import Caching_processor
# Pour l'interface entre les deux
import pandas as pd
import pickle
import json
import os
from math import isnan

# Variables d'environnement
DATADIR = os.getenv("DATADIR")
GENERATED_DIR = os.getenv("GENERATED_DIR")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION")

# L'ensemble des demandes enregistrées
app_train = pd.read_parquet(os.path.join(DATADIR, "application_train_smaller.parquet"))
app_train.drop("TARGET", axis=1, inplace=True) # TARGET n'est pas un input du modèle
app_test = pd.read_parquet(os.path.join(DATADIR, "application_test_smaller.parquet"))
applications = pd.concat([app_train, app_test], axis=0)

# Tables secondaires
prev_app_path = os.path.join(DATADIR, "previous_application_smaller.parquet")
bureau_path = os.path.join(DATADIR, "bureau_smaller.parquet")

# Chargement du modèle
model_prefix = f"./models/{MODEL_NAME}_v{MODEL_VERSION}"
with open(f"{model_prefix}.pickle", "rb") as f:
    pipeline = pickle.load(f)
with open(f"{model_prefix}.json", "rt") as f:
    additional_model_data = json.load(f)
threshold = additional_model_data["threshold"]

# Reconfiguration
processor:Caching_processor = pipeline.named_steps["processor"]
processor.reset_paths(GENERATED_DIR, prev_app_path, bureau_path)
processor._load_secondary_tables()

# Fonctions internes
def clean_up_nans(features_dict:dict)->dict:
    for key, value in features_dict.items():
        if isnan(value):
            features_dict[key] = None
    return features_dict

# API

# Entrées 
min_sk_id = applications["SK_ID_CURR"].min()
max_sk_id = applications["SK_ID_CURR"].max()
class App_ID(BaseModel):
    """Entrée limitée à l'ID de la demande"""
    sk_id_curr : int = Field(ge=min_sk_id, le=max_sk_id, description="ID de la demande à traiter")

class Application_data(BaseModel):
    """
    Informations concernant la demande à traiter 
    """
    SK_ID_CURR:int = Field(ge=min_sk_id, le=max_sk_id, description="ID of loan in our sample. Used to get info about previous applications and credits in other institutions")
    NAME_CONTRACT_TYPE:bool = Field(description="Identification if loan is cash (False) or revolving (True)")
    CODE_GENDER:int = Field(ge=0, le=2, description="Gender of the client. 0:'F', 1:'M', 2:'XNA'")
    FLAG_OWN_CAR:bool = Field(description="Flag if the client owns a car")
    AMT_INCOME_TOTAL:float = Field(ge=0, description="Income of the client")
    AMT_CREDIT:float = Field(ge=0, description="Credit amount of the loan")
    AMT_ANNUITY:Optional[float] = Field(ge=0, description="Loan annuity")
    NAME_EDUCATION_TYPE:int = Field(ge=0, le=4, description="Level of highest education the client achieved. 0:'Lower secondary', 1:'Secondary / secondary special', 2:'Incomplete higher', 3:'Higher education', 4:'Academic degree'")
    NAME_FAMILY_STATUS:int = Field(ge=0, le=5, description="Family status of the client. 0:'Married', 1:'Single / not married', 2:'Civil marriage', 3:'Separated', 4:'Widow', 5:'Unknown'")
    DAYS_BIRTH:int = Field(gt=0, description="Client's age in days at the time of application (time only relative to the application)")
    DAYS_EMPLOYED:int = Field(description="How many days before the application the person started current employment (time only relative to the application). Should be positive. Negative values are allowed for compatibility reasons but will be treated as missing.")
    DAYS_ID_PUBLISH:int = Field(ge=0, description="How many days before the application did client change the identity document with which he applied for the loan (time only relative to the application)")
    EXT_SOURCE_1:Optional[float] = Field(ge=0, le=1, description="Normalized score from external data source (normalized)")
    EXT_SOURCE_2:Optional[float] = Field(ge=0, le=1, description="Normalized score from external data source (normalized)")
    EXT_SOURCE_3:Optional[float] = Field(ge=0, le=1, description="Normalized score from external data source (normalized)")
    DAYS_LAST_PHONE_CHANGE:int = Field(ge=0, description="How many days before application did client change phone")

app_predict = FastAPI(
    title="API de prédiction du risque de retard de paiement",
    description="""
API de prédiction du risque de retard de paiement

Cette API fournit des prédictions du risque qu'un demandeur de prêt se retrouve en retard de paiement.

**Types de requêtes disponibles**
 - */get_application_data* : à partir de l'ID d'une demande, récupère dans la base de donnée les 
 informations sur la demande pertinentes pour effectuer une prédiction.
 - */predict* : prédiction du risque par un modèle de machine learning, sous la forme d'une décision
   de rejet (true=demande à rejeter) et d'une probabilité de retard de paiement (proche de 1=risque élevé)

**Utilisation**
- Récupérez les données sur la demande à traiter avec */get_application_data* .
- Passez ces données à */predict* pour obtenir une prédiction.
L'intérêt d'effectuer la prédiction en deux temps est de permettre à l'utilisateur d'étudier l'impact des paramètres sur la prédiction
"""
)

@app_predict.get("/")
def root():
    """informations de base"""
    return {
        "message" : "Application basique de prédiction de probabilité de retard de paiement",
        "status" : "running",
        "available_endpoints": {
            "get data": "/get_application_data",
            "prédiction": "/predict",
            "docs": "/docs",
        },
    }

@app_predict.post("/get_application_data")
def get_application_data(input_data:App_ID):
    """
    À partir de l'ID d'une demande, récupère dans la base de donnée les informations sur la demande
    pertinentes pour effectuer une prédiction
    """
    features = applications[applications["SK_ID_CURR"] == input_data.sk_id_curr]
    if features.shape[0] == 0:
        raise HTTPException(500, f"Provided ID ({input_data.sk_id_curr}) not in internal database")
    features_dict = features.loc[features.index[0]].to_dict()
    features_dict = clean_up_nans(features_dict)
    return features_dict

@app_predict.post("/predict")
def predict_default_risk(input_data:Application_data):
    """
    Prédiction du risque par un modèle de machine learning, sous la forme d'une décision de rejet
    (true=demande à rejeter) et d'une probabilité de retard de paiement (proche de 1=risque élevé)
    """
    features = pd.DataFrame(input_data.model_dump(), index=[0])
    proba = pipeline.predict_proba(features)[0,1]
    pred = (proba >= threshold)
    proba = float(proba)
    pred = bool(pred)
    return {"prediction":pred, "probability":proba}