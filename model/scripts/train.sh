#!/usr/bin/env bash

INPUT=$1
NOW="$(date +"%Y%m%d_%H%M%S")"
JOB_PREFIX="hashy"

JOB_NAME="${JOB_PREFIX}_train_${NOW}"
JOB_DIR="gs://buffer-temp/jobs/${JOB_NAME}"
PACKAGE_PATH=trainer
MAIN_TRAINER_MODULE=$PACKAGE_PATH.task
REGION=us-central1


CMD="gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --config config/train_config.yaml \
  --job-dir $JOB_DIR \
  --stream-logs \
  --package-path $PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  -- \
  --input $INPUT \
  --workers 48
  "

echo $CMD
eval $CMD
