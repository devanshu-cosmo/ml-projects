import numpy as np
import string
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from hangman.features.feature_extractor import HangmanFeatureExtractor

class MaskedLetterModelTrainer:

    #def __init__(self, dictionary):
        #self.dictionary = dictionary
        #self.fx = HangmanFeatureExtractor(dictionary)
        #self.model = None
        
    def __init__(self, dictionary_path):
        self.dictionary_path = dictionary_path

        with open(dictionary_path, "r") as f:
            self.dictionary = f.read().splitlines()

        self.fx = HangmanFeatureExtractor(dictionary_path)
        self.model = None

    def make_dataset(self, n_words=200000, masks_per_word=2):
        X, y = [], []
        words = np.random.choice(self.dictionary, size=min(n_words, len(self.dictionary)), replace=False)

        for w in words:
            L = len(w)
            if L < 3:
                continue

            for _ in range(masks_per_word):
                pos = np.random.randint(0, L)
                target = w[pos]

                clean = list(w)
                clean[pos] = "_"              # mask exactly one position
                clean = "".join(clean)

                feats = self.fx.extract_maskpos_features(clean, pos)
                X.append(feats)
                y.append(target)

        return X, np.array(y)

    def train(self, X_dict, y):
        self.model = Pipeline([
            ("vec", DictVectorizer(sparse=True)),
            ("clf", LogisticRegression(
                multi_class="multinomial",
                solver="saga",
                max_iter=250,
                n_jobs=1
            ))
        ])
        self.model.fit(X_dict, y)
        return self.model

    def save(self, path="mask_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load(path="mask_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
