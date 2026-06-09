import argparse
import json
import os
import re
import time
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional

import pandas as pd
import requests
from tqdm import tqdm


# =========================
# 1. Basic utilities
# =========================

REQUIRED_COLUMNS = [
    "original_id",
    "candidate_id",
    "original_text",
    "candidate_text",
]


def normalize_col(col: str) -> str:
    col = str(col).strip().lower()
    col = col.replace(" ", "_").replace("-", "_")
    col = col.replace("parapharse", "paraphrase")
    return col


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def extract_generator_from_filename(path: Path) -> str:
    """
    Example:
    aya_llm_judge.csv -> aya
    deepseek_llm_judge.csv -> deepseek
    """
    name = path.stem.lower()
    name = name.replace("_llm_judge", "")
    name = name.replace("-llm-judge", "")
    return name


def read_input_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={c: normalize_col(c) for c in df.columns})

    # Drop index-like columns
    drop_cols = [c for c in df.columns if c.startswith("unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Fallback aliases
    if "original_text" not in df.columns and "text" in df.columns:
        df["original_text"] = df["text"]

    if "candidate_text" not in df.columns:
        possible_candidate_cols = [
            "paraphrase",
            "paraphrase_r1",
            "paraphrase_r2",
            "paraphrase_r3",
            "generated_text",
            "output_text",
        ]

        for col in possible_candidate_cols:
            if col in df.columns:
                df["candidate_text"] = df[col]
                break

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")

    df["original_text"] = df["original_text"].apply(clean_text)
    df["candidate_text"] = df["candidate_text"].apply(clean_text)

    if "source" not in df.columns:
        df["source"] = "unknown"

    df["generator"] = extract_generator_from_filename(path)

    return df


def load_all_inputs(input_dir: str, pattern: str = "*_llm_judge.csv") -> pd.DataFrame:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No input files found in {input_dir} with pattern {pattern}")

    dfs = []

    for file in files:
        print(f"Loading: {file.name}")
        dfs.append(read_input_file(file))

    df = pd.concat(dfs, ignore_index=True)

    # Unique global id for judge result
    df["judge_item_id"] = (
        df["generator"].astype(str)
        + "::"
        + df["candidate_id"].astype(str)
    )

    # Remove duplicated judge items
    df = df.drop_duplicates(subset=["judge_item_id"]).reset_index(drop=True)

    return df


# =========================
# 2. Rule-based pre-check
# =========================

AI_ARTIFACT_PATTERNS = [
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bi am an ai\b",
    r"\bi cannot\b",
    r"\bi can’t\b",
    r"tôi là (một )?(ai|trí tuệ nhân tạo)",
    r"là (một )?mô hình ngôn ngữ",
    r"với tư cách là (một )?(ai|trí tuệ nhân tạo)",
    r"as an artificial intelligence",
]


ENCODING_NOISE_PATTERNS = [
    r"�",
    r"Ã.",
    r"Â.",
    r"áº",
    r"à¡",
    r"ðŸ",
]


def sequence_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def regex_any(patterns: List[str], text: str) -> bool:
    text = text.lower()
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def rule_based_precheck(row: pd.Series) -> Dict[str, Any]:
    original = row["original_text"]
    candidate = row["candidate_text"]

    flags = {
        "empty_original": original == "",
        "empty_candidate": candidate == "",
        "ai_artifact_flag": regex_any(AI_ARTIFACT_PATTERNS, candidate),
        "encoding_noise_flag": regex_any(ENCODING_NOISE_PATTERNS, candidate),
        "too_short_flag": len(candidate.split()) < 2,
        "too_similar_flag": sequence_similarity(original, candidate) >= 0.95,
    }

    hard_drop = (
        flags["empty_original"]
        or flags["empty_candidate"]
        or flags["ai_artifact_flag"]
        or flags["encoding_noise_flag"]
    )

    return {
        **flags,
        "surface_similarity": round(sequence_similarity(original, candidate), 4),
        "rule_predecision": "drop" if hard_drop else "judge",
    }


# =========================
# 3. Prompt builder
# =========================

def build_judge_prompt(row: pd.Series) -> str:
    return f"""
You are a strict data quality judge for an AI-generated text detection dataset on social media.

Your task is to evaluate whether the candidate text is a valid generated/paraphrased sample based on the original human-written text.

Context:
- The dataset is for AI-generated text detection on social media.
- The text may be short, informal, slangy, emotional, or grammatically imperfect.
- Do not penalize informal social-media style.
- Penalize robotic, overly formal, template-like, or AI-disclaimer content.
- Penalize candidates that change the original meaning.
- Penalize candidates that are too broken, unreadable, or in the wrong language.

Original human-written text:
"{row["original_text"]}"

Candidate generated/paraphrased text:
"{row["candidate_text"]}"

Metadata:
- source: {row.get("source", "unknown")}
- generator: {row.get("generator", "unknown")}
- candidate_id: {row.get("candidate_id", "unknown")}
- bleu_score: {row.get("bleu_score", "N/A")}
- bertscore: {row.get("bertscore", "N/A")}
- edit_similarity: {row.get("edit_similarity", "N/A")}
- surface_similarity: {row.get("surface_similarity", "N/A")}

Score each criterion from 1 to 5:

1 = very poor
2 = poor
3 = acceptable but questionable
4 = good
5 = excellent

Criteria:
1. meaning_score:
   Does the candidate preserve the meaning, intent, stance, or emotion of the original text?

2. naturalness_score:
   Does the candidate sound fluent and natural?

3. social_style_score:
   Does the candidate fit social-media writing style instead of sounding like an essay or formal article? Social-media writing style must include these characteristics: short, informal, slangy, emotional, or grammatically imperfect. Do not penalize for social-media style. Penalize if it sounds like an essay, news, or formal writing.

4. language_consistency_score:
   Does the candidate use the same language as the original text?

5. ai_artifact_score:
   5 means no AI artifact.
   1 means obvious AI artifact, disclaimer, template phrase, or robotic wording.

6. dataset_usefulness_score:
   Is this candidate useful for training an AI-generated text detector without introducing obvious shortcut bias?

Decision rule:
- "keep": high-quality and useful sample.
- "review": usable but questionable, should be manually checked.
- "drop": low-quality, wrong meaning, wrong language, obvious AI artifact, broken text, or harmful shortcut sample.

Return only valid JSON. Do not include markdown.

JSON schema:
{{
  "meaning_score": 1,
  "naturalness_score": 1,
  "social_style_score": 1,
  "language_consistency_score": 1,
  "ai_artifact_score": 1,
  "dataset_usefulness_score": 1,
  "overall_quality_score": 1,
  "decision": "keep",
  "reason": "short reason"
}}
""".strip()


# =========================
# 4. LLM Judge client
# =========================

from huggingface_hub import InferenceClient


class HFJudgeClient:
    def __init__(
        self,
        model: str,
        hf_token: str,
        provider: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 120,
        json_mode: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode

        self.client = InferenceClient(
            token=hf_token,
            provider=provider,
            timeout=timeout,
        )

    def judge(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a strict data quality judge. Always return valid JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat_completion(**kwargs)

        return response.choices[0].message.content

# =========================
# 5. JSON parsing
# =========================

def parse_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object found in response: {text[:300]}")

    return json.loads(match.group(0))


def normalize_judge_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    score_keys = [
        "meaning_score",
        "naturalness_score",
        "social_style_score",
        "language_consistency_score",
        "ai_artifact_score",
        "dataset_usefulness_score",
        "overall_quality_score",
    ]

    clean = {}

    for key in score_keys:
        value = obj.get(key, None)

        try:
            value = int(value)
        except Exception:
            value = None

        if value is not None:
            value = max(1, min(5, value))

        clean[key] = value

    decision = str(obj.get("decision", "review")).lower().strip()

    if decision not in ["keep", "review", "drop"]:
        decision = "review"

    clean["decision"] = decision
    clean["reason"] = str(obj.get("reason", "")).strip()

    return clean


# =========================
# 6. Final decision combiner
# =========================

def combine_final_decision(row: Dict[str, Any]) -> str:
    """
    Combine rule-based flags + LLM decision.
    This makes the pipeline safer than trusting LLM output alone.
    """

    if row.get("rule_predecision") == "drop":
        return "drop"

    if row.get("ai_artifact_flag") is True:
        return "drop"

    if row.get("encoding_noise_flag") is True:
        return "drop"

    if row.get("empty_candidate") is True:
        return "drop"

    llm_decision = row.get("decision", "review")

    meaning = row.get("meaning_score")
    language = row.get("language_consistency_score")
    artifact = row.get("ai_artifact_score")
    usefulness = row.get("dataset_usefulness_score")
    overall = row.get("overall_quality_score")

    if meaning is not None and meaning <= 2:
        return "drop"

    if language is not None and language <= 2:
        return "drop"

    if artifact is not None and artifact <= 2:
        return "drop"

    if usefulness is not None and usefulness <= 2:
        return "drop"

    if overall is not None and overall < 3:
        return "drop"

    # Too similar is not always wrong, but should be reviewed
    if row.get("too_similar_flag") is True:
        return "review"

    return llm_decision


# =========================
# 7. Resume support
# =========================

def load_completed_ids(jsonl_path: Path) -> set:
    if not jsonl_path.exists():
        return set()

    completed = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                completed.add(obj["judge_item_id"])
            except Exception:
                continue

    return completed


def append_jsonl(path: Path, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================
# 8. Main pipeline
# =========================

def run_pipeline(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_jsonl = output_dir / "judge_results.jsonl"
    result_csv = output_dir / "judge_results.csv"
    summary_csv = output_dir / "quality_summary_by_generator.csv"
    accepted_csv = output_dir / "accepted_candidates.csv"

    df = load_all_inputs(args.input_dir, args.pattern)

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    print(f"Total candidates loaded: {len(df)}")

    # Rule-based precheck
    precheck_rows = []

    for _, row in df.iterrows():
        precheck_rows.append(rule_based_precheck(row))

    precheck_df = pd.DataFrame(precheck_rows)
    df = pd.concat([df.reset_index(drop=True), precheck_df], axis=1)

    hf_token = os.getenv(args.hf_token_env, "")

    if not hf_token:
        raise ValueError(
            f"Missing Hugging Face token. Please set environment variable: {args.hf_token_env}"
        )

    client = HFJudgeClient(
        model=args.model,
        hf_token=hf_token,
        provider=args.hf_provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        json_mode=args.json_mode,
    )

    completed_ids = load_completed_ids(result_jsonl) if args.resume else set()

    print(f"Already completed: {len(completed_ids)}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        row_dict = row.to_dict()
        judge_item_id = row_dict["judge_item_id"]

        if judge_item_id in completed_ids:
            continue

        base_result = {
            "judge_item_id": judge_item_id,
            "original_id": row_dict.get("original_id"),
            "candidate_id": row_dict.get("candidate_id"),
            "generator": row_dict.get("generator"),
            "source": row_dict.get("source"),
            "original_text": row_dict.get("original_text"),
            "candidate_text": row_dict.get("candidate_text"),
            "surface_similarity": row_dict.get("surface_similarity"),
            "empty_original": row_dict.get("empty_original"),
            "empty_candidate": row_dict.get("empty_candidate"),
            "ai_artifact_flag": row_dict.get("ai_artifact_flag"),
            "encoding_noise_flag": row_dict.get("encoding_noise_flag"),
            "too_short_flag": row_dict.get("too_short_flag"),
            "too_similar_flag": row_dict.get("too_similar_flag"),
            "rule_predecision": row_dict.get("rule_predecision"),
        }

        # If rule-based hard drop, we can skip LLM call to save cost
        if row_dict.get("rule_predecision") == "drop" and args.skip_rule_drop:
            result = {
                **base_result,
                "meaning_score": None,
                "naturalness_score": None,
                "social_style_score": None,
                "language_consistency_score": None,
                "ai_artifact_score": None,
                "dataset_usefulness_score": None,
                "overall_quality_score": None,
                "decision": "drop",
                "final_decision": "drop",
                "reason": "Dropped by rule-based precheck.",
                "raw_judge_response": None,
                "judge_error": None,
            }

            append_jsonl(result_jsonl, result)
            continue

        prompt = build_judge_prompt(row)

        raw_response = None
        judge_error = None
        parsed = None

        for attempt in range(args.max_retries):
            try:
                raw_response = client.judge(prompt)
                parsed = normalize_judge_output(parse_json_from_text(raw_response))
                break

            except Exception as e:
                judge_error = str(e)
                wait_time = 2 ** attempt
                time.sleep(wait_time)

        if parsed is None:
            parsed = {
                "meaning_score": None,
                "naturalness_score": None,
                "social_style_score": None,
                "language_consistency_score": None,
                "ai_artifact_score": None,
                "dataset_usefulness_score": None,
                "overall_quality_score": None,
                "decision": "review",
                "reason": "Judge failed or returned invalid JSON.",
            }

        merged = {
            **base_result,
            **parsed,
            "raw_judge_response": raw_response,
            "judge_error": judge_error,
        }

        merged["final_decision"] = combine_final_decision(merged)

        append_jsonl(result_jsonl, merged)

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Convert JSONL to CSV
    results = []

    with open(result_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    result_df = pd.DataFrame(results)
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    # Summary
    summary = (
        result_df
        .groupby("generator")
        .agg(
            total=("judge_item_id", "count"),
            keep=("final_decision", lambda x: (x == "keep").sum()),
            review=("final_decision", lambda x: (x == "review").sum()),
            drop=("final_decision", lambda x: (x == "drop").sum()),
            avg_overall_quality=("overall_quality_score", "mean"),
            avg_meaning=("meaning_score", "mean"),
            avg_naturalness=("naturalness_score", "mean"),
            avg_social_style=("social_style_score", "mean"),
            avg_ai_artifact=("ai_artifact_score", "mean"),
        )
        .reset_index()
    )

    summary["keep_rate"] = summary["keep"] / summary["total"]
    summary["review_rate"] = summary["review"] / summary["total"]
    summary["drop_rate"] = summary["drop"] / summary["total"]

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # Accepted candidates
    accepted = result_df[result_df["final_decision"] == "keep"].copy()
    accepted.to_csv(accepted_csv, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Result JSONL: {result_jsonl}")
    print(f"Result CSV: {result_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Accepted candidates: {accepted_csv}")


# =========================
# 9. CLI
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/llm_judge")
    parser.add_argument("--pattern", default="*_llm_judge.csv")

    parser.add_argument("--model", required=True)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--hf-provider",
        default=None,
        help="Optional Hugging Face inference provider, e.g. hf-inference, together, sambanova, fireworks-ai. Leave empty for default routing."
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0)

    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-rule-drop", action="store_true")

    parser.add_argument("--max-rows", type=int, default=None)

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()