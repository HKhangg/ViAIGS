# evaluation/tokenizer.py
import os
import pandas as pd
from underthesea import word_tokenize
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


def tokenize(text, backend="underthesea", language="vi"):
    """
    Tokenize văn bản tiếng Việt.

    backend options:
        - "underthesea" (mặc định): dùng underthesea, tương thích Python 3.12
        - "vncorenlp":              dùng VnCoreNLP, cần set VNCORENLP_DIR
        - "polyglot":               legacy, KHÔNG dùng trên Python 3.12
    """
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

    # Mặc định: underthesea
    return word_tokenize(text, format="text").split()