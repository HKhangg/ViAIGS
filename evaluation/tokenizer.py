# evaluation/tokenizer.py

import os
import pandas as pd
from polyglot.text import Text
from vncorenlp import VnCoreNLP

_VNCORENLP_SEGMENTER = None


def load_vncorenlp_segmenter():
    global _VNCORENLP_SEGMENTER

    if _VNCORENLP_SEGMENTER is not None:
        return _VNCORENLP_SEGMENTER

    vncorenlp_dir = os.getenv("VNCORENLP_DIR", "").strip()
    if not vncorenlp_dir:
        raise ValueError("VNCORENLP_DIR is not set.")

    _VNCORENLP_SEGMENTER = VnCoreNLP(
        vncorenlp_dir,
        annotators="wseg",
        max_heap_size="-Xmx2g",
    )
    return _VNCORENLP_SEGMENTER


def tokenize(text, backend="polyglot", language="vi"):
    text = "" if pd.isna(text) else str(text).strip()
    if not text:
        return []

    if backend == "vncorenlp":
        segmenter = load_vncorenlp_segmenter()
        sentences = segmenter.tokenize(text)
        return [
            token.replace("_", " ")
            for sentence in sentences
            for token in sentence
        ]

    return [str(token) for token in Text(text, hint_language_code=language).words]