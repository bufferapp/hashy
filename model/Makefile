.DEFAULT_GOAL := run

IMAGE_NAME := gcr.io/buffer-data/hashy-model:latest
GCLOUD_CONFIG_FLAG = -v $(HOME)/.config/gcloud/:/root/.config/gcloud

.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

.PHONY: data
data:
	@bash scripts/get_dataset.sh

.PHONY: run
run: build
	docker run -it --rm $(GCLOUD_CONFIG_FLAG) $(IMAGE_NAME) bash scripts/train.sh data/updates_hashtags.csv

.PHONY: dev
dev: build
	docker run -it $(GCLOUD_CONFIG_FLAG) -v $(PWD):/model --rm $(IMAGE_NAME) /bin/bash

.PHONY: push
push: build
	docker push $(IMAGE_NAME)
