# -------- Définition de la distribution Python ----------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# --------------------- Configurations UV -------------------------
# Permet aux packages installés d'être compilés et importés plus rapidement dans le code
ENV UV_COMPILE_BYTECODE=1
# Copie depuis le cache au lieu de créer des liens car c'est un volume monté
ENV UV_LINK_MODE=copy
# Permet à UV d'attendre plus longtemps pour installer les packages, en cas de projet avec beaucoup de dépendances
ENV UV_HTTP_TIMEOUT=1000


# # ------ Installation des packages
# Copie uniquement les fichiers nécessaires pour reproduire l'environnement virtuel avec les packages
COPY pyproject.toml uv.lock /app/
# Installe les dépendances du projet en utilisant le lockfile et les paramètres, pour une vitesse maximale
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group ui

# Activation de l'environnement virtuel
ENV PATH="/app/.venv/bin:$PATH"

# LightGBM a besoin de bibliothèques en C++ qui ne sont bien évidemment pas incluses dans la release pypi
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1



# -------- Définition des variables d'environnement ---------
ARG APP_PORT
ARG HF_BUCKET_URL
ARG LOGGING_PERIOD

# Utilisé uniquement dans ce fichier 
ENV APP_PORT=$APP_PORT
# Utilisé par api.py
ENV HF_BUCKET_URL=$HF_BUCKET_URL
ENV LOGGING_PERIOD=$LOGGING_PERIOD

# # Secrets
RUN --mount=type=secret,id=HF_BUCKET_TOKEN,mode=0444,required=false \
   if [ -f /run/secrets/HF_BUCKET_TOKEN ]; then (cat /run/secrets/HF_BUCKET_TOKEN > /app/secret_HF_BUCKET_TOKEN); fi

# ------- Transfert des fichiers --------
COPY ./data /app/data/
COPY ./models /app/models/
COPY ./.streamlit/config.toml /app/.streamlit/config.toml
COPY ./src/* /app/
COPY ./parallel-run-api-ui.sh /app/
COPY ./ui_assets/* /app/ui_assets/
RUN mkdir logs

# Si on veut seulement déployer une API sans UI, décommenter le bloc ci-dessous et commenter les blocs suivants
# (Le paramètre de build --target n'est pas disponible sur Hugging Face)
# # ------ Lancement de l'API 
# EXPOSE $APP_PORT
# # Cette forme permet de paramétrer la commande exécutée avec des variables d'environnement sans casser la transmission normale des signaux
# SHELL ["/bin/sh", "-c"]
# CMD exec uvicorn api:app_predict --host 0.0.0.0 --port $APP_PORT

# ------ Paramétrage de Streamlit
RUN echo "port = $APP_PORT" >> .streamlit/config.toml

# ------ Lancement de l'API et de l'UI en parallèle
EXPOSE $APP_PORT
SHELL ["/bin/bash", "-c"]
CMD . ./parallel-run-api-ui.sh