. ./envars.sh
docker build -f Dockerfile -t dockerized_model_api --build-arg API_PORT="$API_PORT" .