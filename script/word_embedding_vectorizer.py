import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class WordEmbeddingVectorizer(object):
    def __init__(self, word_model):
        self.word_model = word_model
        self.word_idf_weight = None
        self.vector_size = word_model.wv.vector_size

    def fit(self, texts):
        text_docs = []
        for text in texts:
            text_docs.append(" ".join(text))

        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
        tfidf.fit(text_docs)
        max_idf = max(tfidf.idf_)
        self.word_idf_weight = defaultdict(lambda: max_idf,
                                           [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, texts):
        text_word_vector = self.word_average_list(texts)
        return text_word_vector

    def word_average(self, sent):
        mean = []
        for word in sent:
            if word in self.word_model.wv.vocab:
                mean.append(self.word_model.wv.get_vector(
                    word) * self.word_idf_weight[word])

        if not mean:
            logging.warning(
                "cannot compute average owing to no vector for {}".format(sent))
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def word_average_list(self, docs):
        return np.vstack([self.word_average(sent) for sent in docs])
