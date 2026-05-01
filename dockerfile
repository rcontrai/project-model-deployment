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
    uv sync --locked --no-dev

# Activation de l'environnement virtuel
ENV PATH="/app/.venv/bin:$PATH"

# LightGBM a besoin de bibliothèques en C++ qui ne sont bien évidemment pas incluses dans la release pypi
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1



# -------- Définition des variables d'environnement ---------
ARG MODEL_NAME
ARG MODEL_VERSION

ENV MODEL_NAME=$MODEL_NAME
ENV MODEL_VERSION=$MODEL_VERSION

# ------- Transfert des fichiers --------
COPY ./data /app/data/
COPY ./models /app/models/
COPY ./src/* /app/

# ------ Lancement de l'API 
EXPOSE 8000
CMD ["uvicorn", "app:app_predict", "--host", "0.0.0.0", "--port", "8000"]