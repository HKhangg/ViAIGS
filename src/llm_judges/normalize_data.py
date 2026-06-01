import argparse
import json
import re
from pathlib import Path

import pandas as pd


def read_table(input_path: str) -> pd.DataFrame:
    """
    Read CSV / Excel file.
    """
    path = Path(input_path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def normalize_column_name(col: str) -> str:
    """
    Normalize column names:
    - lowercase
    - strip spaces
    - replace spaces with underscores
    - fix common typo: parapharse -> paraphrase
    """
    col = str(col).strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace("parapharse", "paraphrase")
    return col


def clean_text(text):
    """
    Basic text cleaning for judge input.
    Keep this light. Do not over-clean social media text.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove excessive spaces but keep natural punctuation/emojis.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_paraphrase_columns(columns):
    """
    Detect paraphrase columns automatically.
    Supports:
    - paraphrase
    - paraphrase_1
    - paraphrase_r1
    - paraphrase.1
    - parapharse_r1
    """
    paraphrase_cols = []

    for col in columns:
        normalized = normalize_column_name(col)

        if normalized.startswith("paraphrase"):
            paraphrase_cols.append(col)

    return paraphrase_cols


def extract_round_number(col_name: str, default_idx: int) -> int:
    """
    Extract paraphrase round number from column name.
    Examples:
    - paraphrase_r1 -> 1
    - paraphrase_2 -> 2
    - paraphrase.3 -> 3
    """
    name = normalize_column_name(col_name)

    match = re.search(r"(?:r|_|\.)(\d+)$", name)
    if match:
        return int(match.group(1))

    return default_idx


def safe_get(row, col_name, default=None):
    if col_name in row.index:
        return row[col_name]
    return default


# =========================
# 2. Main processing
# =========================

def convert_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide paraphrase format into long candidate format.

    Input format example:
        id, text, paraphrase_r1, paraphrase_r2, paraphrase_r3, source, length, bleu_score...

    Output format:
        original_id, candidate_id, original_text, candidate_text,
        source, paraphrase_round, length, scores...
    """

    # Normalize actual dataframe column names
    original_columns = list(df.columns)
    normalized_map = {col: normalize_column_name(col) for col in original_columns}
    df = df.rename(columns=normalized_map)

    # Drop unnamed columns if they only represent dataframe index
    unnamed_cols = [c for c in df.columns if c.startswith("unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    # Required columns
    if "id" not in df.columns:
        raise ValueError("Missing required column: id")

    if "text" not in df.columns:
        raise ValueError("Missing required column: text")

    paraphrase_cols = detect_paraphrase_columns(df.columns)

    if not paraphrase_cols:
        raise ValueError(
            "No paraphrase columns found. Expected columns like paraphrase_r1, paraphrase_r2, paraphrase_r3."
        )

    rows = []

    for _, row in df.iterrows():
        original_id = str(row["id"])
        original_text = clean_text(row["text"])
        source = safe_get(row, "source", "unknown")

        for idx, para_col in enumerate(paraphrase_cols, start=1):
            candidate_text = clean_text(row[para_col])

            if not candidate_text:
                continue

            paraphrase_round = extract_round_number(para_col, idx)

            candidate_id = f"{original_id}_para_r{paraphrase_round}"

            item = {
                "original_id": original_id,
                "candidate_id": candidate_id,
                "original_text": original_text,
                "candidate_text": candidate_text,
                "source": source,
                "generation_method": "paraphrase",
                "paraphrase_round": paraphrase_round,
                "candidate_length": len(candidate_text.split()),
            }

            # Keep all metric columns if available
            metric_cols = [
                "length",
                "bleu_score",
                "bertscore",
                "crossencoder_score",
                "cross_encoder_score",
                "ld",
                "levenshtein_distance",
                "edit_similarity",
            ]

            for metric in metric_cols:
                if metric in df.columns:
                    item[metric] = row[metric]

            rows.append(item)

    long_df = pd.DataFrame(rows)

    return long_df


# =========================
# 3. Build LLM Judge prompt
# =========================

def build_judge_prompt(row: pd.Series) -> str:
    """
    Build a structured prompt for LLM Judge.
    The judge should evaluate whether the generated social-media text is usable.
    """

    prompt = f"""
You are a data quality judge for an AI-generated text detection dataset on social media.

Your task is to evaluate whether the candidate text is a good generated/paraphrased sample based on the original human-written text.

Evaluate the candidate using these criteria:

1. Meaning Preservation:
- Does the candidate preserve the meaning or intent of the original text?

2. Naturalness:
- Does the candidate sound natural and fluent?

3. Social Media Style:
- Does the candidate match informal social-media writing style?
- It should not be too formal, robotic, or essay-like.

4. Language Consistency:
- Does the candidate use the same language as the original text?

5. AI Artifact:
- Does the candidate contain obvious AI artifacts such as "As an AI language model", explanations, disclaimers, or unnatural generic wording?

6. Dataset Usefulness:
- Is this candidate useful for training an AI-generated text detector?

Original text:
{row["original_text"]}

Candidate text:
{row["candidate_text"]}

Metadata:
- source: {row.get("source", "unknown")}
- paraphrase_round: {row.get("paraphrase_round", "unknown")}
- bleu_score: {row.get("bleu_score", "N/A")}
- bertscore: {row.get("bertscore", "N/A")}
- edit_similarity: {row.get("edit_similarity", "N/A")}
- levenshtein_distance: {row.get("ld", row.get("levenshtein_distance", "N/A"))}

Return only valid JSON with this schema:

{{
  "meaning_score": 1-5,
  "naturalness_score": 1-5,
  "social_style_score": 1-5,
  "language_consistency_score": 1-5,
  "ai_artifact_score": 1-5,
  "overall_quality_score": 1-5,
  "decision": "keep" | "review" | "drop",
  "reason": "short explanation"
}}
""".strip()

    return prompt


def export_llm_judge_jsonl(long_df: pd.DataFrame, output_path: str):
    """
    Export JSONL file for LLM Judge.
    Each line is one candidate sample.
    """

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in long_df.iterrows():
            record = {
                "candidate_id": row["candidate_id"],
                "original_id": row["original_id"],
                "source": row.get("source", "unknown"),
                "original_text": row["original_text"],
                "candidate_text": row["candidate_text"],
                "prompt": build_judge_prompt(row),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# 4. CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare paraphrase dataset for LLM Judge."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV/XLSX file in wide paraphrase format."
    )

    parser.add_argument(
        "--output-csv",
        default="quality_candidates_long.csv",
        help="Output CSV file in long format."
    )

    parser.add_argument(
        "--output-jsonl",
        default="llm_judge_input.jsonl",
        help="Output JSONL file for LLM Judge."
    )

    args = parser.parse_args()

    df = read_table(args.input)

    long_df = convert_wide_to_long(df)

    long_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    export_llm_judge_jsonl(long_df, args.output_jsonl)

    print("Done.")
    print(f"Long-format CSV saved to: {args.output_csv}")
    print(f"LLM Judge JSONL saved to: {args.output_jsonl}")
    print(f"Total judge samples: {len(long_df)}")


if __name__ == "__main__":
    main()