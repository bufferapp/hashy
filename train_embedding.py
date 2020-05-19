from gensim.models import Word2Vec
from gensim.models import KeyedVectors

w2vmodel = Word2Vec(
    corpus_file="data/updates_hashtags.csv",
    size=400,
    window=8,
    iter=30,
    min_count=30,
    workers=16,
)

word_vectors = w2vmodel.wv

word_vectors.save("app/vectors/updates_hashtags_vectors.kv")
