from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

w2vmodel = Word2Vec(
    corpus_file="updates_hashtag.ls",
    size=500,
    window=5,
    iter=20,
    min_count=30,
    workers=16,
)

print(w2vmodel.wv.most_similar("#smile"))

w2vmodel.save("updates_hashtags_word2vec.model")

word_vectors = w2vmodel.wv

word_vectors.save("updates_hashtags_vectors.kv")
