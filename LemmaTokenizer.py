from sklearn.base import TransformerMixin
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer(TransformerMixin):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def transform(self, X):
        return [" ".join([self.wnl.lemmatize(word) for word in doc.split()]) for doc in X]

    def fit(self):
        return self
