.DEFAULT_GOAL := run

IMAGE_NAME := gcr.io/buffer-data/hashy-model:latest
GCLOUD_CONFIG_FLAG = -v $(HOME)/.config/gcloud/:/root/.config/gcloud

.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .


.PHONY: run
run: build
	docker run -it --rm $(IMAGE_NAME)

.PHONY: dev
dev:
	docker run -it $(GCLOUD_CONFIG_FLAG) -v $(PWD):/model --rm $(IMAGE_NAME) /bin/bash

.PHONY: push
push: build
	docker push $(IMAGE_NAME)

