---
title: Project Model Deployment
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
short_description: https://github.com/rcontrai/project-model-deployment
---

# project-model-deployment

Mise en oeuvre du projet "Confirmez vos compétences en MLOps (Partie 2/2)" du bootcamp "Déployez vos modèles de machine learning" de OpenClassrooms.

Ce projet est développé sur [GitHub](https://github.com/rcontrai/project-model-deployment) puis déployé sur [Hugging Face](https://huggingface.co/spaces/rcontrai/project-model-deployment). Il faut donc se fier à GitHub pour l'historique de développement et de déploiement.

Ce dépôt met à disposition une API (interne pour la version en ligne) et une UI de prédiction du risque de retard de paiement à partir du modèle développé dans la partie 1 du projet. L'interface utilisateur est découplée du reste du code afin d'être facilement remplaçable.

## Fonctionnement de l'API

Les endpoints principaux de l'API sont */get_application_data* qui permet d'obtenir les caractéristiques d'une demande de la base de données à partir de son ID, et */predict* qui renvoie la décision et le score prédit par le modèle à partir des caractéristiques d'une demande. Tous deux sont des requêtes POST. Ils disposent de schémas d'entrée et de sortie qui s'assurent de la conformité de ce qu'ils envoient et reçoivent.

D'autres endpoints fournissent des informations statiques utiles à l'UI.

Pour le détail des endpoints et des schémas d'entrée et de sortie, se fier à la documentation disponible sous [localhost:8000/docs](http://localhost:8000/docs) quand l'API tourne en local.

## Installation

### Image docker

Les scripts sont faits pour s'éxécuter à l'intérieur d'un conteneur. [build-docker.sh](/build-docker.sh) et [run-docker.sh](/run-docker.sh) contiennent des exemples de commandes à utiliser pour construire l'image en local. Ils ont besoin d'un script envars.sh qui définit les variables d'environnement utilisées dans les commandes. Les variables d'environnement sont les suivantes :

| Variable  | Description |
| ------------- | ------------- |
| APP_PORT  | port utilisé par l'application  |
| LOGGING_PERIOD  | (optionnel) période de synchronisation des logs en secondes (défaut 21600) |
| HF_BUCKET_URL | (optionnel) chemin du bucket huggingface où uploader les logs |
| HF_BUCKET_TOKEN | (optionnel) token d'accès au bucket |

L'application est disponible sur localhost:APP_PORT\
Le Dockerfile peut facilement être modifié pour mettre à disposition l'API plutôt que l'UI (se fier aux commentaires).

### Code local

L'environnement s'installe avec uv :\
`uv sync --group ui --locked`

Pour éxécuter l'API en local :\
`uv run uvicorn --app-dir=./src api:app_predict --host 0.0.0.0 --port 8000`

Pour éxécuter l'UI en local (nécéssite d'éxécuter l'API en même temps sur le port 8000):\
`uv run streamlit run src/ui.py`

## Structure du dépôt

### code

*src/* contient le code de l'application. [api.py](src/api.py) contient le code de l'API et [ui.py](src/ui.py) le code de l'UI. Les autres fichiers décrivent la préparation des données utilisée par le modèle.

*.github/* définit le pipeline CI/CD, qui lance les tests automatiques à chaque pull request et qui envoie le code sur le space huggingface à chaque push sur la branche *main*.

*logs/* contient le script [export_logs](/logs/export_logs.py) qui permet de télécharger les logs (si HF_BUCKET_TOKEN est set) et de les exporter au format parquet. C'est aussi là que les logs sont enregistrés en local.

*tests/* contient les tests automatiques (à exécuter avec pytest).

*update/* contient des scripts qui aident à la mise à jour du dépôt vers une nouvelle version du modèle. Lire [le README du dossier](/update/README.md) pour plus de détails.

### assets

*.streamlit/* config de streamlit.

*data/* données utilisées pour l'inférence.

*models/* le modèle à déployer au format pickle, avec des informations supplémentaires au format json.

*ui_assets/* assets utilisés par l'UI