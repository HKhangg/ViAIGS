# evaluation/metric.py

from collections import Counter
import editdistance
from nltk.translate.meteor_score import meteor_score


def build_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def ngram_f1(ref_tokens, cand_tokens, n=3):
    n = min(n, len(ref_tokens), len(cand_tokens))
    if n == 0:
        return 0.0

    ref_ngrams = Counter(build_ngrams(ref_tokens, n))
    cand_ngrams = Counter(build_ngrams(cand_tokens, n))

    overlap = sum((ref_ngrams & cand_ngrams).values())

    precision = overlap / max(sum(cand_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)

    if precision + recall == 0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 4)


def meteor_score_vi(ref_tokens, cand_tokens):
    try:
        return round(meteor_score([ref_tokens], cand_tokens), 4)
    except:
        return 0.0


def edit_similarity(ref, cand):
    dist = editdistance.eval(ref, cand)
    max_len = max(len(ref), len(cand), 1)
    return dist, round(1 - dist / max_len, 4)