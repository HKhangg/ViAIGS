# evaluation/evaluator.py

import numpy as np
from tqdm import tqdm
from evaluate import load

from .tokenizer import tokenize
from .metric import (
    meteor_score_vi,
    ngram_f1,
    edit_similarity
)


class Evaluator:
    def __init__(self, tokenizer_backend="polyglot"):
        self.tokenizer_backend = tokenizer_backend
        self.bertscore = load("bertscore")

    def compute_bertscore(self, refs, cands):
        try:
            results = self.bertscore.compute(
                predictions=cands,
                references=refs,
                model_type="bert-base-multilingual-cased",
            )
            return [round(float(x), 4) for x in results["f1"]]
        except:
            return [0.0] * len(refs)

    def evaluate_dataframe(self, df):
        refs = df["text"].astype(str).tolist()
        cands = df["paraphrase"].astype(str).tolist()

        bert_scores = self.compute_bertscore(refs, cands)

        meteors = []
        ngrams = []
        edit_dists = []
        edit_sims = []

        for ref, cand in tqdm(zip(refs, cands), total=len(refs), desc="Evaluating"):
            ref_tokens = tokenize(ref, backend=self.tokenizer_backend)
            cand_tokens = tokenize(cand, backend=self.tokenizer_backend)

            meteors.append(meteor_score_vi(ref_tokens, cand_tokens))
            ngrams.append(ngram_f1(ref_tokens, cand_tokens))
            
            dist, sim = edit_similarity(ref, cand)
            edit_dists.append(dist)
            edit_sims.append(sim)

        df["meteor"] = meteors
        df["bertscore"] = bert_scores
        df["ngram"] = ngrams
        df["editdistance"] = edit_dists
        df["edit_similarity"] = edit_sims

        return df

    def summarize(self, df):
        return {
            "samples": len(df),
            "meteor_mean": round(df["meteor"].mean(), 4),
            "bertscore_mean": round(df["bertscore"].mean(), 4),
            "ngram_mean": round(df["ngram"].mean(), 4),
            "editdistance_mean": round(df["editdistance"].mean(), 4),
            "edit_similarity_mean": round(df["edit_similarity"].mean(), 4),
        }