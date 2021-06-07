#!/usr/bin/env bash

cd "$(dirname $0)"

set -e

. ./settings.sh

echo "Running training pipeline"

cd "$APP_DIR"

echo "Creating models directory"
mkdir -p "${MODELS_DIR}"

echo "Creating pipe directory"
mkdir -p "${PIPE_DIR}"

echo "Executing training"
python main/train.py \
       --source ${TRAINING_DATA_PATH} \
       --pipe_path ${PIPE_DIR} \
       --model_path ${MODELS_DIR}

echo "Training executed successfully!"