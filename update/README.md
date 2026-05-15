# /update

Ce dossier contient des scripts qui ne sont pas voués à être déployés, mais qui servent à mettre à jour certains assets comme le modèle ou les graphes d'illustration utilisés par l'UI.

Ils n'utilisent pas les mêmes dépendances que le reste du projet, car ils ont été conçus dans un environnement de R&D avec des dépendances qui seraient superflues en production. Installation : `pip install -r update/requirements.txt`

*export_registered_model.py* télécharge un modèle depuis un serveur mlfow et l'enregistre en .pickle.

*generate_graphs.py* pré-génère les graphes et autres assets utilisés par l'UI à partir de l'ensemble d'entraînement et du modèle.