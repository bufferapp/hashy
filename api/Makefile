.DEFAULT_GOAL := run

IMAGE_NAME := gcr.io/buffer-data/hashy:latest

.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .


.PHONY: run
run: build
	docker run -it -p 80:80 --rm $(IMAGE_NAME)

.PHONY: train
train:
	docker run -it -v $(PWD):/app --rm $(IMAGE_NAME) python train_embedding.py

.PHONY: dev
dev:
	docker run -it -v $(PWD):/app -p 80:80 --rm $(IMAGE_NAME) /bin/bash

.PHONY: push
push: build
	docker push $(IMAGE_NAME)

.PHONY: deploy
deploy: push
	gcloud run deploy hashy --image $(IMAGE_NAME) --platform managed --region us-central1
