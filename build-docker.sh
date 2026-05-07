. ./envars.sh
docker build -f Dockerfile -t dockerized_model_api --build-arg APP_PORT="$APP_PORT" .