#!/bin/env bash

export APP_DIR=/usr/src/app

################################################################
########################## TRAINING ############################
################################################################

export MODELS_DIR=${APP_DIR}/main/trained_models/

export PIPE_DIR=${APP_DIR}/main/trained_pipe/

export TRAINING_DATA_PATH=${APP_DIR}/main/data/spam.csv


echo "Setting environment variables."
echo "============================="
env
echo "============================="