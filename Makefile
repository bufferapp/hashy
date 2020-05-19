.DEFAULT_GOAL := run

IMAGE_NAME := gcr.io/buffer-data/hashy:latest

.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

.PHONY: push
push: build
	docker push $(IMAGE_NAME)

.PHONY: run
run:
	docker run -it -p 80:80 --rm $(IMAGE_NAME)

.PHONY: data
data:
	@bash scripts/get_dataset.sh

.PHONY: train
train:
	docker run -it -v $(PWD):/app --rm $(IMAGE_NAME) python train_embedding.py

.PHONY: dev
dev:
	docker run -it -v $(PWD):/app -p 80:80 --rm $(IMAGE_NAME) /bin/bash
