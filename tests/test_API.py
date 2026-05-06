# Tests de l'API

from fastapi.testclient import TestClient
from pytest import approx
from src.api import app_predict, applications, pipeline, threshold, model_prediction

client = TestClient(app_predict)
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

def test_get_application_data_ok():
    for sk_id_curr in test_examples:
        response = client.post("/get_application_data", json={"sk_id_curr":str(sk_id_curr)})
        assert response.status_code == 200

def test_get_application_data_nok():
    too_small = applications["SK_ID_CURR"].min() - 1
    response = client.post("/get_application_data", json={"sk_id_curr":str(too_small)})
    assert response.status_code == 422
    too_big = applications["SK_ID_CURR"].max() + 1
    response = client.post("/get_application_data", json={"sk_id_curr":str(too_big)})
    assert response.status_code == 422
    response = client.post("/get_application_data", json={"sk_id_curr":"text"})
    assert response.status_code == 422

def test_predict_ok():
    for sk_id_curr in test_examples:
        get_appdata_response = client.post("/get_application_data", json={"sk_id_curr":str(sk_id_curr)})
        predict_body = get_appdata_response.json()
        predict_response = client.post("/predict", json=predict_body)
        assert predict_response.status_code == 200
        predict_response_body = predict_response.json()
        assert predict_response_body["prediction"] is not None
        assert predict_response_body["probability"] is not None

def test_predict_nok():
    get_appdata_response = client.post("/get_application_data", json={"sk_id_curr":str(test_examples[0])})
    base_predict_body:dict = get_appdata_response.json()
    nok_values = {
        "SK_ID_CURR": [int(applications["SK_ID_CURR"].min() - 1), int(applications["SK_ID_CURR"].max() + 1), "text", None],
        "NAME_CONTRACT_TYPE": [0.5, "text", None],
        "CODE_GENDER": [-1, 3, 0.5, "text", None],
        "FLAG_OWN_CAR": [0.5, "text", None],
        "AMT_INCOME_TOTAL": [-1., "text", None],
        "AMT_CREDIT": [-1., 0., "text", None],
        "AMT_ANNUITY": [-1., 0., "text"],
        "NAME_EDUCATION_TYPE": [-1, 5, 0.5, "text", None],
        "NAME_FAMILY_STATUS": [-1, 6, 0.5, "text", None],
        "DAYS_BIRTH": [-1, 0, 0.5, "text", None],
        "DAYS_EMPLOYED": [0.5, "text", None],
        "DAYS_ID_PUBLISH": [-1, 0.5, "text", None],
        "EXT_SOURCE_1": [-0.5, 1.5, "text"],
        "EXT_SOURCE_2": [-0.5, 1.5, "text"],
        "EXT_SOURCE_3": [-0.5, 1.5, "text"],
        "DAYS_LAST_PHONE_CHANGE": [-1, 0.5, "text", None]
    }
    for key in base_predict_body.keys():
        for nok_value in nok_values[key]:
            predict_body = base_predict_body.copy()
            predict_body[key] = nok_value
            predict_response = client.post("/predict", json=predict_body)
            assert predict_response.status_code == 422

def test_predict_consistent():
    for sk_id_curr in test_examples:
        # Résultat calculé avec l'API
        get_appdata_response = client.post("/get_application_data", json={"sk_id_curr":str(sk_id_curr)})
        predict_body = get_appdata_response.json()
        predict_response = client.post("/predict", json=predict_body)
        predict_response_body = predict_response.json()
        # Résultat calculé en local
        features_in = applications[applications["SK_ID_CURR"] == sk_id_curr]
        pred, proba = model_prediction(pipeline, threshold, features_in)
        assert bool(predict_response_body["prediction"]) == pred
        assert float(predict_response_body["probability"]) == approx(proba)