from gensim.models import Word2Vec
from gensim.models import KeyedVectors

w2vmodel = Word2Vec(
    corpus_file="data/updates_hashtags_full.csv",
    size=100,
    window=5,
    iter=10,
    min_count=50,
    workers=16,
)

word_vectors = w2vmodel.wv

word_vectors.save("app/vectors/updates_hashtags_vectors.kv")
