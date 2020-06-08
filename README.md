# Hashy

![GitHub](https://img.shields.io/github/license/bufferapp/hashy?style=flat-square)

Hashtag recommendations for Instagram using Machine Learning.

## Quickstart

The models expect a file in [LineSentence](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.LineSentence) format. Each line should be a set of hashtags separated by a whitespace. E.g:

```
#data #github #ml
#cats #dogs #instagram
```

The [`model`](./model) folder contains the code to train and deploy a model using Google Cloud AI Platform.

The [`api`](./api) folder contains the code to train an embedding and serve a basic API using FastAPI.
