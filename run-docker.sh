. ./envars.sh
# # Image qui expose l'API pour tester l'UI plus vite
docker run -p "8000":"8000" dockerized_model_api
# # Image qui contient l'UI
# docker run -p "$APP_PORT":"$APP_PORT" dockerized_model_ui