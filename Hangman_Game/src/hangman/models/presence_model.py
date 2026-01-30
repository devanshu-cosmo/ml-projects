import pickle
import numpy as np

from hangman.features.feature_extractor import HangmanFeatureExtractor


class PresenceModel:
    def __init__(self, model, feature_extractor):
        self.model = model
        self.fx = feature_extractor

    @classmethod
    def load(cls, model_path, dictionary_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        fx = HangmanFeatureExtractor(dictionary_path)
        return cls(model, fx)

    def score_letter(self, word_state, guessed, wrong, letter):
        feats = self.fx.extract_presence_features(
            word_state, guessed, wrong, letter
        )
        proba = self.model.predict_proba([feats])[0, 1]
        return proba
