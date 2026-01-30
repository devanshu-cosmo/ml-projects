import numpy as np
import random
import collections
from collections import Counter

class HangmanFeatureExtractor:
    def __init__(self, dictionary_path):
        self.dictionary_path = dictionary_path
        self.words = self._load_dictionary(dictionary_path)
        self.dictionary = self.words
        self._precompute_statistics()
    #def __init__(self, dictionary):
        #self.dictionary = dictionary
        #self._precompute_statistics()

    def _load_dictionary(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f]


    def _precompute_statistics(self):
        # 1) Global letter frequency
        self.global_letter_freq = collections.Counter("".join(self.dictionary))
        self.global_total = sum(self.global_letter_freq.values())

        # 2) Letter frequency by word length
        self.length_letter_freq = {}
        self.length_total = {}
        for w in self.dictionary:
            L = len(w)
            if L not in self.length_letter_freq:
                self.length_letter_freq[L] = collections.Counter()
                self.length_total[L] = 0
            self.length_letter_freq[L].update(w)
            self.length_total[L] += len(w)

        # 3) Position letter distributions by length
        self.position_letter_freq = {}
        self.position_total = {}
        for w in self.dictionary:
            L = len(w)
            if L not in self.position_letter_freq:
                self.position_letter_freq[L] = [collections.Counter() for _ in range(L)]
                self.position_total[L] = [0 for _ in range(L)]
            for i, ch in enumerate(w):
                self.position_letter_freq[L][i][ch] += 1
                self.position_total[L][i] += 1

        # 4) Co-occurrence
        self.letter_cooccurrence = {}
        for w in self.dictionary:
            u = set(w)
            for a in u:
                if a not in self.letter_cooccurrence:
                    self.letter_cooccurrence[a] = collections.Counter()
                for b in u:
                    if a != b:
                        self.letter_cooccurrence[a][b] += 1

        # 5) Bigram + trigram stats with boundaries
        # padded: ^^word$$ so left2/left1/right1/right2 always exist
        self.bigram_prev = collections.Counter()   # (prev, cur)
        self.bigram_next = collections.Counter()   # (cur, next)
        self.prev_total = collections.Counter()    # prev -> *
        self.out_total = collections.Counter()     # cur -> *
        self.trigram = collections.Counter()       # (prev, cur, next)
        self.tri_context_total = collections.Counter()  # (prev, next) -> *

        for w in self.dictionary:
            p = "^^" + w + "$$"
            for i in range(2, 2 + len(w)):  # points to letters in original word
                prev = p[i - 1]
                cur = p[i]
                nxt = p[i + 1]

                self.bigram_prev[(prev, cur)] += 1
                self.bigram_next[(cur, nxt)] += 1
                self.prev_total[prev] += 1
                self.out_total[cur] += 1

                self.trigram[(prev, cur, nxt)] += 1
                self.tri_context_total[(prev, nxt)] += 1

    def _max_consecutive_blanks(self, clean_word):
        best = 0
        cur = 0
        for c in clean_word:
            if c == "_":
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    def _char_at(self, clean_word, idx):
        if idx < 0 or idx >= len(clean_word):
            return None
        c = clean_word[idx]
        return c if c != "_" else None

    def extract_presence_features(self, word_state, guessed_letters, wrong_letters, candidate_letter):
        """
        Features for binary 'presence' model: P(candidate_letter is in the word).
        candidate-free; uses only priors + context from the masked pattern.
        """
        feats = {}

        clean = word_state[::2]
        L = len(clean)
        blanks = [i for i, c in enumerate(clean) if c == "_"]
        revealed = [c for c in clean if c != "_"]

        feats["word_length"] = L
        feats["num_blanks"] = len(blanks)
        feats["num_revealed"] = len(revealed)
        feats["completion_rate"] = (len(revealed) / L) if L else 0.0

        feats["num_guessed"] = len(guessed_letters)
        feats["num_wrong"] = len(wrong_letters)
        feats["wrong_vowel_count"] = sum(1 for x in wrong_letters if x in "aeiou")

        feats["is_vowel"] = 1 if candidate_letter in "aeiou" else 0
        feats["blank_cluster_size"] = self._max_consecutive_blanks(clean)
        feats["has_blank_start"] = 1 if blanks and blanks[0] == 0 else 0
        feats["has_blank_end"] = 1 if blanks and blanks[-1] == L - 1 else 0

        # Global + length priors (normalized)
        g = self.global_letter_freq.get(candidate_letter, 0)
        feats["global_p"] = g / max(1, self.global_total)

        lf = self.length_letter_freq.get(L, collections.Counter()).get(candidate_letter, 0)
        feats["length_p"] = lf / max(1, self.length_total.get(L, 1))

        # Position prior over blank positions: average P(letter at pos)
        pos_ps = []
        if L in self.position_letter_freq:
            for pos in blanks:
                denom = self.position_total[L][pos] if pos < len(self.position_total[L]) else 0
                num = self.position_letter_freq[L][pos].get(candidate_letter, 0) if pos < len(self.position_letter_freq[L]) else 0
                pos_ps.append(num / max(1, denom))
        feats["blank_pos_p_mean"] = float(np.mean(pos_ps)) if pos_ps else 0.0
        feats["blank_pos_p_max"] = float(np.max(pos_ps)) if pos_ps else 0.0

        # Co-occurrence with revealed letters
        co = 0
        if candidate_letter in self.letter_cooccurrence:
            for r in revealed:
                co += self.letter_cooccurrence[candidate_letter].get(r, 0)
        feats["cooccurrence_score"] = co
        feats["cooccurrence_avg"] = co / max(1, len(revealed))

        # Context n-grams: aggregate over blank positions where neighbor(s) are known
        prev_probs = []
        next_probs = []
        tri_probs = []

        for pos in blanks:
            left1 = self._char_at(clean, pos - 1)
            right1 = self._char_at(clean, pos + 1)

            if left1 is not None:
                num = self.bigram_prev.get((left1, candidate_letter), 0)
                den = self.prev_total.get(left1, 0)
                prev_probs.append(num / max(1, den))

            if right1 is not None:
                num = self.bigram_next.get((candidate_letter, right1), 0)
                den = self.out_total.get(candidate_letter, 0)
                next_probs.append(num / max(1, den))

            if left1 is not None and right1 is not None:
                num = self.trigram.get((left1, candidate_letter, right1), 0)
                den = self.tri_context_total.get((left1, right1), 0)
                tri_probs.append(num / max(1, den))

        feats["ctx_prev_p_max"] = float(np.max(prev_probs)) if prev_probs else 0.0
        feats["ctx_next_p_max"] = float(np.max(next_probs)) if next_probs else 0.0
        feats["ctx_tri_p_max"] = float(np.max(tri_probs)) if tri_probs else 0.0

        feats["ctx_prev_p_mean"] = float(np.mean(prev_probs)) if prev_probs else 0.0
        feats["ctx_next_p_mean"] = float(np.mean(next_probs)) if next_probs else 0.0
        feats["ctx_tri_p_mean"] = float(np.mean(tri_probs)) if tri_probs else 0.0

        return feats

    def extract_maskpos_features(self, clean_word, pos):
        """
        Features for masked-letter multiclass model at ONE blank position:
        predicts which letter fits at position pos.
        """
        L = len(clean_word)
        padded = "^^" + clean_word + "$$"
        # Map pos in clean_word to pos+2 in padded
        i = pos + 2

        # Use '_' for unknown neighbors so the model can learn "no context" too
        left2 = padded[i - 2] if padded[i - 2] != "_" else "_"
        left1 = padded[i - 1] if padded[i - 1] != "_" else "_"
        right1 = padded[i + 1] if padded[i + 1] != "_" else "_"
        right2 = padded[i + 2] if padded[i + 2] != "_" else "_"

        feats = {
            "L": L,
            "pos": pos,
            "pos_norm": pos / max(1, L - 1),
            "left2": left2, "left1": left1,
            "right1": right1, "right2": right2,
            "is_start": int(pos == 0),
            "is_end": int(pos == L - 1),
        }
        return feats
