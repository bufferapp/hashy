#!/usr/bin/env bash

INPUT=$1
NOW="$(date +"%Y%m%d_%H%M%S")"
JOB_PREFIX="hashy"

JOB_NAME="${JOB_PREFIX}_hptuning_${NOW}"
JOB_DIR="gs://buffer-temp/jobs/"
PACKAGE_PATH=trainer
MAIN_TRAINER_MODULE=$PACKAGE_PATH.task
REGION=us-central1


CMD="gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --config config/hptuning_config.yaml \
  --job-dir $JOB_DIR \
  --stream-logs \
  --package-path $PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  -- \
  --input $INPUT
  "

echo $CMD
eval $CMD
