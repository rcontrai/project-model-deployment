# -------- Définition de la distribution Python ----------
FROM python:3.12-slim

WORKDIR /app

# ------ Installation des packages
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

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