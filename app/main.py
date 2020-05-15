
from gensim.models import KeyedVectors
from fastapi import FastAPI
from collections import Counter
from string import punctuation

import spacy
import en_core_web_sm

# Load small POS model
nlp = en_core_web_sm.load()

# Load embedding vectors
word_vectors = KeyedVectors.load("models/updates_hashtags_vectors.kv", mmap="r")

app = FastAPI()


@app.get("/")
def main():
    return {"Hello": "World"}


@app.post("/hashtagify")
def get_hashtags(input: dict = None):
    result = []
    pos_tag = ["PROPN", "ADJ", "NOUN"]
    doc = nlp(input["text"].lower())
    for token in doc:
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            result.append(token.text)

    return [("#" + x[0]) for x in Counter(result).most_common(5)]

@app.get("/similar/{hashtag}")
def get_similar_hashtags(hashtag: str):
    try:
        return [i[0] for i in word_vectors.most_similar("#" + hashtag.lower())]
    except Exception as e:
        return "Hashtag not in vocabulary"
