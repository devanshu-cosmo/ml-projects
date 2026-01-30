import pickle
import numpy as np

from hangman.features.feature_extractor import HangmanFeatureExtractor


class MaskedLetterModel:
    def __init__(self, model, feature_extractor):
        self.model = model
        self.fx = feature_extractor

    @classmethod
    def load(cls, model_path, dictionary_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        fx = HangmanFeatureExtractor(dictionary_path)
        return cls(model, fx)

    def score_letter(self, word_state, pos, letter):
        feats = self.fx.extract_maskpos_features(word_state, pos)

        # predict_proba returns probabilities in class order
        proba = self.model.predict_proba([feats])[0]

        # map letter -> probability
        classes = self.model.classes_
        if letter in classes:
            idx = np.where(classes == letter)[0][0]
            return proba[idx]
        else:
            return 0.0
