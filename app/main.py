from collections import Counter
from fastapi import FastAPI
from gensim.models import KeyedVectors
from string import punctuation
from pydantic import BaseModel

import en_core_web_sm
import spacy


class Caption(BaseModel):
    text: str


# Load small POS model
nlp = en_core_web_sm.load()

# Load embedding vectors
word_vectors = KeyedVectors.load("vectors/updates_hashtags_vectors.kv", mmap="r")

# Start API
app = FastAPI()


@app.get("/")
def main():
    return {"Hello": "World"}


@app.post("/hashtagify")
def hashtagify(caption: Caption):
    result = []
    pos_tag = ["PROPN", "ADJ", "NOUN"]
    doc = nlp(caption.text.lower())
    for token in doc:
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            result.append(token.text)

    return [("#" + x[0]) for x in Counter(result).most_common(5)]


@app.get("/similar/{hashtag}")
def similar(hashtag: str):
    try:
        return [i[0] for i in word_vectors.most_similar("#" + hashtag.lower())]
    except Exception as e:
        return "Hashtag not in vocabulary"
