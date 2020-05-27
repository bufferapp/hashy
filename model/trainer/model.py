"""ML model definitions."""
from sklearn.base import BaseEstimator
from gensim.models import Word2Vec


class Word2VecEstimator(BaseEstimator):
    def __init__(self, size=50, window=5, min_count=10, compute_loss=True):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.compute_loss = compute_loss
        self.loss = None

    def fit(self, X, y=None):

        self.model = Word2Vec(
            sentences=X,
            size=self.size,
            window=self.window,
            min_count=self.min_count,
            compute_loss=self.compute_loss,
        )

        self.loss = -self.model.get_latest_training_loss()
        self.model = self.model.wv
        return self

    def predict(self, X, y=None, topn=10):
        return [similar[0] for similar in self.model.most_similar(X)]

    def score(self, X=None, y=None):
        return self.loss
