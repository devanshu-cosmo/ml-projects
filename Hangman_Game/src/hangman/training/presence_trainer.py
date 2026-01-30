import numpy as np
import string
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from hangman.features.feature_extractor import HangmanFeatureExtractor

class PresenceModelTrainer:
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


    def _make_state_from_letters(self, word, revealed_letters):
        # realistic hangman state: reveal all occurrences of guessed letters
        return " ".join([ch if ch in revealed_letters else "_" for ch in word])

    def make_dataset(self, n_words=150000, states_per_word=3, max_wrong=2):
        X, y = [], []
        words = np.random.choice(self.dictionary, size=min(n_words, len(self.dictionary)), replace=False)

        alphabet = list(string.ascii_lowercase)
        for w in words:
            uniq = set(w)
            L = len(w)
            if L < 3:
                continue

            for _ in range(states_per_word):
                # revealed letters: choose some letters from the word
                k = np.random.randint(0, max(1, min(len(uniq), 4)))
                revealed = set(np.random.choice(list(uniq), size=k, replace=False)) if k > 0 else set()

                # wrong letters: sample from outside the word
                outside = [c for c in alphabet if c not in uniq]
                kw = np.random.randint(0, max_wrong + 1)
                wrong = set(np.random.choice(outside, size=min(kw, len(outside)), replace=False)) if kw > 0 else set()

                guessed = set(revealed) | set(wrong)
                word_state = self._make_state_from_letters(w, revealed)

                for cand in alphabet:
                    if cand in guessed:
                        continue
                    feats = self.fx.extract_presence_features(word_state, guessed, wrong, cand)
                    X.append(feats)
                    y.append(1 if cand in uniq else 0)

        return X, np.array(y, dtype=np.int32)

    def train(self, X_dict, y):
        self.model = Pipeline([
            ("vec", DictVectorizer(sparse=True)),
            ("clf", LogisticRegression(
                solver="saga",
                max_iter=250,
                n_jobs=1,
                class_weight="balanced"
            ))
        ])
        self.model.fit(X_dict, y)
        return self.model

    def save(self, path="presence_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load(path="presence_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
