# -------- Définition de la distribution Python ----------
FROM python:3.12-slim

WORKDIR /app

# ------ Installation des packages
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# -------- Définition des variables d'environnement ---------
ARG DATADIR
ARG GENERATED_DIR
ARG MODEL_NAME
ARG MODEL_VERSION

ENV DATADIR=$DATADIR
ENV GENERATED_DIR=$GENERATED_DIR
ENV MODEL_NAME=$MODEL_NAME
ENV MODEL_VERSION=$MODEL_VERSION

# ------- Transfert des fichiers --------
COPY ./src/* /app/
COPY ./models /app/

# ------ Lancement de l'API 
EXPOSE 8000
CMD ["uvicorn", "app:app_predict", "--host", "0.0.0.0", "--port", "8000"]