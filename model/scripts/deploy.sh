#!/usr/bin/env bash

# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Convenience script for deploying trained scikit-learn model.
#
# Prerequisites:
#   - Google Cloud SDK
#
# Globals:
#   PROJECT_ID: Google Cloud project to use.
#
# Arguments:
#   $1: Path to directory containing trained and exported scikit-learn model
#   $2: Name of the model to be deployed
#   $3: Version of the model to be deployed

MODEL_DIR=$1
MODEL_NAME=$2
VERSION_NAME=$3

REGION=us-central1
FRAMEWORK=scikit-learn
RUN_TIME=1.15
PYTHON_VERSION=3.7
STAGING_BUCKET="gs://buffer-temp"
PACKAGE_URI="gs://buffer-temp/models/hashy/hashy-0.1.0.tar.gz"

python setup.py sdist --formats=gztar

gsutil cp ./dist/hashy-0.1.0.tar.gz "$PACKAGE_URI"

if gcloud ai-platform models list | grep "$MODEL_NAME" &> /dev/null
then
   echo "Model already exists."
else
    # 1. Create model
    gcloud ai-platform models create "$MODEL_NAME" \
    --regions=$REGION
fi


if gcloud ai-platform versions list --model="$MODEL_NAME" | grep "$VERSION_NAME" &> /dev/null
then
   echo "Version already exists."
else
    # 2. Create version
    gcloud --quiet beta ai-platform versions create "$VERSION_NAME" \
    --model "$MODEL_NAME" \
    --origin "$MODEL_DIR" \
    --staging-bucket "$STAGING_BUCKET" \
    --runtime-version="$RUN_TIME" \
    --python-version="$PYTHON_VERSION" \
    --framework "$FRAMEWORK" \
    --machine-type mls1-c4-m2 \
    --package-uris="$PACKAGE_URI"
fi

# echo "#lunch" | gcloud ai-platform predict --model hashy --version beta --text-instances -
