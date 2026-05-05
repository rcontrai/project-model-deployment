. ./envars.sh
docker build -f dockerfile -t dockerized_model_api --build-arg API_PORT="$API_PORT" .