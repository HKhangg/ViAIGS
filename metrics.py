import os
import sys
from collections import Counter
from pathlib import Path

import editdistance
import numpy as np
import pandas as pd
from evaluate import load
from nltk.translate.meteor_score import meteor_score
from polyglot.text import Text
from tqdm import tqdm


if len(sys.argv) < 3:
    raise SystemExit(
        "Usage: python metrics.py <model_or_model1,model2> <dataset_dir> [tokenizer_backend]"
    )


MODELS = [model.strip() for model in sys.argv[1].split(",") if model.strip()]
DATASET_DIR = Path(sys.argv[2])
TOKENIZER_BACKEND = sys.argv[3].strip().lower() if len(sys.argv) > 3 else "polyglot"


np.random.seed(42)
bertscore = load("bertscore")
_VNCORENLP_SEGMENTER = None


def load_vncorenlp_segmenter():
    global _VNCORENLP_SEGMENTER

    if _VNCORENLP_SEGMENTER is not None:
        return _VNCORENLP_SEGMENTER

    vncorenlp_dir = os.getenv("VNCORENLP_DIR", "").strip()
    if not vncorenlp_dir:
        raise ValueError("VNCORENLP_DIR is not set.")

    from vncorenlp import VnCoreNLP

    _VNCORENLP_SEGMENTER = VnCoreNLP(
        vncorenlp_dir,
        annotators="wseg",
        max_heap_size="-Xmx2g",
    )
    return _VNCORENLP_SEGMENTER


def custom_tokenizer(text, language="vi", backend=TOKENIZER_BACKEND):
    text = "" if pd.isna(text) else str(text).strip()
    if not text:
        return []

    if backend == "vncorenlp":
        segmenter = load_vncorenlp_segmenter()
        sentences = segmenter.tokenize(text)
        return [token.replace("_", " ") for sentence in sentences for token in sentence]

    return [str(token) for token in Text(text, hint_language_code=language).words]


def get_meteor(df, tokenizer_backend=TOKENIZER_BACKEND):
    metric = [0.0] * len(df)
    for position, row in enumerate(
        tqdm(df.itertuples(index=False), total=len(df), desc="METEOR")
    ):
        reference_tokens = custom_tokenizer(row.text, backend=tokenizer_backend)
        candidate_tokens = custom_tokenizer(row.generated, backend=tokenizer_backend)
        if not reference_tokens or not candidate_tokens:
            continue
        try:
            metric[position] = round(meteor_score([reference_tokens], candidate_tokens), 4)
        except Exception:
            metric[position] = 0.0
    return metric


def get_bertscore(df):
    predictions = df["generated"].astype(str).tolist()
    references = df["text"].astype(str).tolist()
    try:
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            model_type="bert-base-multilingual-cased",
        )
        return [round(float(score), 4) for score in results["f1"]]
    except Exception:
        return [0.0] * len(df)


def build_ngrams(tokens, n):
    return [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]


def ngram_overlap_score(reference_tokens, candidate_tokens, n=3):
    effective_n = min(n, len(reference_tokens), len(candidate_tokens))
    if effective_n == 0:
        return 0.0

    reference_ngrams = Counter(build_ngrams(reference_tokens, effective_n))
    candidate_ngrams = Counter(build_ngrams(candidate_tokens, effective_n))
    overlap = sum((reference_ngrams & candidate_ngrams).values())
    precision = overlap / max(sum(candidate_ngrams.values()), 1)
    recall = overlap / max(sum(reference_ngrams.values()), 1)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def get_ngram(df, n=3, tokenizer_backend=TOKENIZER_BACKEND):
    metric = [0.0] * len(df)
    for position, row in enumerate(
        tqdm(df.itertuples(index=False), total=len(df), desc=f"{n}-gram")
    ):
        reference_tokens = custom_tokenizer(row.text, backend=tokenizer_backend)
        candidate_tokens = custom_tokenizer(row.generated, backend=tokenizer_backend)
        metric[position] = ngram_overlap_score(reference_tokens, candidate_tokens, n=n)
    return metric


def normalized_edit_similarity(reference, candidate):
    max_length = max(len(reference), len(candidate), 1)
    distance = editdistance.eval(reference, candidate)
    return distance, round(1 - (distance / max_length), 4)


def get_editdistance(df):
    distances = [0.0] * len(df)
    normalized_scores = [0.0] * len(df)
    for position, row in enumerate(
        tqdm(df.itertuples(index=False), total=len(df), desc="EditDistance")
    ):
        try:
            distance, normalized_score = normalized_edit_similarity(
                str(row.text),
                str(row.generated),
            )
            distances[position] = distance
            normalized_scores[position] = normalized_score
        except Exception:
            distances[position] = 0.0
            normalized_scores[position] = 0.0
    return distances, normalized_scores


def load_datasets(dataset_dir, models):
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    datasets = {}
    for model in models:
        filename = dataset_dir / f"ViAIGS_{model}.csv"
        if not filename.exists():
            raise FileNotFoundError(f"Input file not found: {filename}")

        temp = pd.read_csv(filename, lineterminator="\n")
        temp.columns = temp.columns.str.strip()
        required_columns = {"text", "generated"}
        missing_columns = required_columns - set(temp.columns)
        if missing_columns:
            raise ValueError(f"{filename} is missing required columns: {sorted(missing_columns)}")

        temp = temp.copy()
        temp.fillna("", inplace=True)
        temp["text"] = temp["text"].astype(str)
        temp["generated"] = temp["generated"].astype(str)
        datasets[model] = temp
    return datasets


def summarize_metrics(df):
    return {
        "samples": len(df),
        "meteor_mean": round(df["meteor"].mean(), 4),
        "bertscore_mean": round(df["bertscore"].mean(), 4),
        "ngram_mean": round(df["ngram"].mean(), 4),
        "editdistance_mean": round(df["editdistance"].mean(), 4),
        "edit_similarity_mean": round(df["edit_similarity"].mean(), 4),
    }


def main():
    datasets = load_datasets(DATASET_DIR, MODELS)

    detailed_outputs = []
    summary_outputs = []

    for model, dataset in datasets.items():
        print(f"Processing {model}")

        dataset = dataset.copy()
        dataset["multi_label"] = model
        dataset["meteor"] = get_meteor(dataset)
        dataset["bertscore"] = get_bertscore(dataset)
        dataset["ngram"] = get_ngram(dataset, n=3)
        dataset["editdistance"], dataset["edit_similarity"] = get_editdistance(dataset)

        detailed_outputs.append(dataset)
        summary_outputs.append({"model": model, **summarize_metrics(dataset)})

    detailed_df = pd.concat(detailed_outputs, ignore_index=True)
    summary_df = pd.DataFrame(summary_outputs)

    detailed_path = DATASET_DIR / "ViAIGS_metrics_detailed.csv"
    summary_path = DATASET_DIR / "ViAIGS_metrics_summary.csv"
    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nMetric summary:")
    print(summary_df.to_string(index=False))
    print(f"\nDetailed metrics saved to: {detailed_path}")
    print(f"Summary metrics saved to: {summary_path}")
    print(f"Tokenizer backend: {TOKENIZER_BACKEND}")


if __name__ == "__main__":
    main()