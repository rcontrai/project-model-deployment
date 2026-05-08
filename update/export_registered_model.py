# Sauvegarde hors de mlflow un modèle chargé depuis le model registry de mlflow
# Le but est de ne pas avoir besoin de faire appel à un serveur mlflow pendant le déploiement
# et de se passer d'une intallation de mlflow dans l'environnement de production

import os
import mlflow
import pickle
import json

# Configuration
MODEL_NAME = "LGBMClassifier-reduced_features"
MODEL_VERSION = "10"
TRACKING_SERVER_URL = "http://localhost:5000" # serveur mlflow (potentiellement distant)
DATADIR = os.path.abspath("../data")
GENERATED_DIR = os.path.abspath("./generated") # pas vraiment utilisé par le code

# Tables secondaires
prev_app_path = os.path.join(DATADIR, "previous_application_smaller.parquet")
bureau_path = os.path.join(DATADIR, "bureau_smaller.parquet")

# Chargement du modèle
mlflow.set_tracking_uri(TRACKING_SERVER_URL)
pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
run_id = mlflow.tracking.MlflowClient(mlflow.get_tracking_uri()).get_model_version(MODEL_NAME, MODEL_VERSION).run_id
run_data = mlflow.get_run(run_id).data
metric_multiplier = float(run_data.tags['metric_multiplier'])
threshold = run_data.metrics["best_threshold"] / metric_multiplier
additional_model_data = {"threshold": threshold}

# Reconfiguration
processor = pipeline.named_steps["processor"]
processor.reset_paths(GENERATED_DIR, prev_app_path, bureau_path)

# Export
model_prefix = f"./models/{MODEL_NAME}_v{MODEL_VERSION}"
with open(f"{model_prefix}.pickle", "wb") as f:
    pickle.dump(pipeline, f)
with open(f"{model_prefix}.json", "wt") as f:
    json.dump(additional_model_data, f)