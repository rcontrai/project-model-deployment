. ./envars.sh
# # Image qui expose l'API (il faut modifier le dockerfile)
# docker build -f Dockerfile -t dockerized_model_api --build-arg APP_PORT="8000" .

# Image qui contient l'UI
docker build -f Dockerfile -t dockerized_model_ui --build-arg APP_PORT="$APP_PORT" .