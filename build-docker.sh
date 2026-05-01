. ./envars.sh
docker build -f dockerfile -t dockerized_model_api --build-arg MODEL_NAME="$MODEL_NAME" --build-arg MODEL_VERSION="$MODEL_VERSION" .