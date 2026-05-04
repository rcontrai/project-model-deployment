. ./envars.sh
docker build -f dockerfile -t dockerized_model_api --build-arg PLACEHOLDER_ENV_VAR="$PLACEHOLDER_ENV_VAR" .