# Tests qui peuvent s'effectuer sans lancer l'API

import math
import pandas as pd
import numpy as np
from src.api import threshold, pipeline, applications, model_prediction


# Tests sur threshold 
def test_threshold():
    
    assert isinstance(threshold, float)
    assert not math.isnan(threshold)
    # valeurs : selon le type de modèle il peut être dans [0,1] ou dans R donc difficile de tester

# Tests sur la base de données
def test_data():
    # Unicité de l'index du DataFrame
    assert not applications.index.duplicated().any()
    # Unicité des clés SK_ID_CURR
    assert not applications["SK_ID_CURR"].duplicated().any()

# Tests sur le préparateur de features et le modèle
test_examples = [ # exemples de test, identifiés par leur SK_ID_CURR
    100001, # exemple basique
    100002, # exemple basique
    100003, # valeur manquante pour EXT_SOURCE_3
    100004, # valeur manquante pour EXT_SOURCE_1
    100377, # valeur manquante pour EXT_SOURCE_2
    155054, # valeur manquante pour AMT_ANNUITY
    100006, # pas de correspondances dans bureau
    100024, # pas de correspondances dans previous_application
]
# Préparateur de features
def test_feature_processor():
    feature_processor = pipeline.named_steps["processor"]
    features_out_list = []
    index_list = []
    for sk_id_curr in test_examples:
        features_in = applications[applications["SK_ID_CURR"] == sk_id_curr]
        features_out = feature_processor.transform(features_in)
        # imputation correcte des valeurs manquantes
        assert not features_out.isna().any().any()
        features_out_list.append(features_out)
        index_list.append(features_in.index[0])
    # cohérence des types
    dtypes = pd.DataFrame([features_out.dtypes for features_out in features_out_list])
    assert (dtypes.nunique() == 1).all()
    # comportement identique que le calcul soit fait en batch (comme en conception) ou un par un (comme en production)
    features_out_stack = pd.concat(features_out_list, axis=0)
    features_in_batch = applications[applications["SK_ID_CURR"].isin(test_examples)] # l'ordre des indices dans test_examples n'est pas l'ordre d'origine des données
    features_in_batch = features_in_batch.reindex(index_list)
    features_out_batch = feature_processor.transform(features_in_batch)
    assert (features_out_stack == features_out_batch).all().all()

# Modèle
def test_model():
    pred_list = []
    proba_list = []
    index_list = []
    for sk_id_curr in test_examples:
        features_in = applications[applications["SK_ID_CURR"] == sk_id_curr]
        pred, proba = model_prediction(pipeline, threshold, features_in)
        assert isinstance(pred, bool)
        assert isinstance(pred, bool)
        assert not math.isnan(proba)
        pred_list.append(pred)
        proba_list.append(proba)
        index_list.append(features_in.index[0])
    # comportement identique que le calcul soit fait en batch (comme en conception) ou un par un (comme en production)
    pred_stack = np.array(pred_list)
    proba_stack = np.array(proba_list)
    features_in_batch = applications[applications["SK_ID_CURR"].isin(test_examples)]
    features_in_batch = features_in_batch.reindex(index_list)
    proba_batch = pipeline.predict_proba(features_in_batch)[:,1] # La fonction model_prediction ne supporte pas les traitements en batch
    pred_batch = (proba_batch >= threshold)
    assert np.all(pred_batch == pred_stack)
    assert np.all(proba_batch == proba_stack)
