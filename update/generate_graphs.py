# Génère des graphes illustrant la densité des scores prédits par le modèle
# ainsi que des tableaux détaillant les pourcentiles de ces scores

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle
import os
import json
# Ajout du dossier src au path parce que le modèle picklé contient des références à des modules à la racine du path
# (feature_engineering_small et data_caching) qui sont déjà copiés dans src
from pathlib import Path
import sys
_base_dir = str(Path(__file__).parent.parent / "src")
if _base_dir not in sys.path:
    sys.path.insert(1, _base_dir)

# Configuration
MODEL_NAME = "LGBMClassifier-reduced_features"
MODEL_VERSION = "10"
DATADIR = os.path.abspath("../data")
GENERATED_DIR = os.path.abspath("./generated") # pas vraiment utilisé par le code
ASSETS_DIR = os.path.abspath("../ui_assets/")

# Données d'entraînement
app_train = pd.read_parquet(os.path.join(DATADIR, "application_train_smaller.parquet"))
target = app_train["TARGET"]
app_train.drop("TARGET", axis=1, inplace=True) # TARGET n'est pas un input du modèle

# Tables secondaires
prev_app_path = os.path.join(DATADIR, "previous_application_smaller.parquet")
bureau_path = os.path.join(DATADIR, "bureau_smaller.parquet")

# Chargement du modèle
model_prefix = f"../models/{MODEL_NAME}_v{MODEL_VERSION}"
with open(f"{model_prefix}.pickle", "rb") as f:
    pipeline:Pipeline = pickle.load(f)
with open(f"{model_prefix}.json", "rt") as f:
    additional_model_data = json.load(f)
threshold = additional_model_data["threshold"]

# Reconfiguration
processor = pipeline.named_steps["processor"]
processor.reset_paths(GENERATED_DIR, prev_app_path, bureau_path)
processor._load_secondary_tables()

# Calcul des scores
proba = pipeline.predict_proba(app_train)[:,1]
proba_neg = proba[~target] # scores des clients sans problème
proba_pos = proba[target]  # scores des clients en retard de paiement

# Calcul et sauvegarde des pourcentiles
percentiles_neg = np.percentile(proba_neg, range(101)) # du pourcentile 0 au pourcentile 100
percentiles_pos = np.percentile(proba_pos, range(101))
with open(os.path.join(ASSETS_DIR, "percentiles.npz"), "wb") as f:
    np.savez_compressed(f, percentiles_neg=percentiles_neg, percentiles_pos=percentiles_pos)

# Génération et sauvegarde des graphes
plt.figure()
sns.kdeplot(proba_neg, clip=(0,1), color="g")
plt.xlim(0,1)
fig_neg = plt.gcf()
with open(os.path.join(ASSETS_DIR, "figure_noissue_distrib.pickle"), "wb") as f:
    pickle.dump(fig_neg, f)
plt.figure()
sns.kdeplot(proba_pos, clip=(0,1), color="r")
plt.xlim(0,1)
fig_pos = plt.gcf()
with open(os.path.join(ASSETS_DIR, "figure_default_distrib.pickle"), "wb") as f:
    pickle.dump(fig_pos, f)