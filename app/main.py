from collections import Counter
from fastapi import FastAPI
from gensim.models import KeyedVectors
from string import punctuation
from pydantic import BaseModel
from typing import List
import itertools

import en_core_web_sm
import spacy


class Caption(BaseModel):
    text: str


class UserHashtags(BaseModel):
    hashtags: List[str] = []

    def generate_similar_hashtags(self):
        similar_hashtags = []
        hashtags = ["#" + s for s in self.hashtags if "#" + s in word_vectors.vocab]
        for c in range(1, len(hashtags) + 1):
            for subset in itertools.combinations(hashtags, c):
                hts = [i[0] for i in word_vectors.most_similar(list(subset))]
                similar_hashtags.extend(hts)

        return list(set(similar_hashtags))


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


@app.post("/similar")
def similar(user_hashtags: UserHashtags):
    return user_hashtags.generate_similar_hashtags()
