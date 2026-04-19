# main.py

import argparse
import json
import yaml
import os
import pandas as pd

from tqdm.auto import tqdm
from builders.model_builder import build_model
from utils.postprocess import clean_generated_text
from evaluation.evaluator import Evaluator
from huggingface_hub import login


# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    return parser.parse_args()


# =========================
# Resume helpers
# =========================
def load_existing(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, sep=";")
    except Exception:
        return None


# =========================
# Main
# =========================
def main():
    args = parse_args()

    # ---- config ----
    with open(args.config_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- HF login ----
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # ---- model ----
    model = build_model(cfg)
    model.load_model()

    # ---- data ----
    with open(cfg["data"]["input"], encoding="utf-8") as f:
        data = json.load(f)

    output_path = cfg["data"]["output"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_rounds = cfg.get("paraphrase", {}).get("num_rounds", 5)

    # =========================
    # Resume
    # =========================
    old_df = load_existing(output_path)

    # Init storage từ data gốc
    data_dict = {
        i: {"id": i, "text": data[i].get("text", "").strip()}
        for i in range(len(data))
    }

    if old_df is not None:
        # Load lại paraphrase cũ
        for _, row in old_df.iterrows():
            i = int(row["id"])
            for c in old_df.columns:
                if c.startswith("paraphrase_r"):
                    data_dict[i][c] = row[c]

        # Tìm round đang dở — round đầu tiên mà có ít nhất 1 sample chưa có giá trị
        start_round = 0
        start_sample = 0
        for r in range(num_rounds):
            col = f"paraphrase_r{r+1}"
            if col not in old_df.columns:
                # Round này chưa có gì → bắt đầu từ đây, sample 0
                start_round = r
                start_sample = 0
                break
            # Round có column nhưng có thể có NaN (crash giữa chừng)
            missing = old_df[col].isna() | (old_df[col].astype(str).str.strip() == "")
            if missing.any():
                start_round = r
                # Bắt đầu từ sample đầu tiên còn thiếu
                start_sample = int(old_df[missing].index[0])
                break
        else:
            # Tất cả round đã đầy đủ → chỉ chạy evaluation
            start_round = num_rounds
            start_sample = 0

        if start_round < num_rounds:
            print(f"Loaded {len(data)} samples | Resume từ Round {start_round + 1}, Sample {start_sample}")
        else:
            print(f"Loaded {len(data)} samples | Inference đã xong, chỉ chạy evaluation")
    else:
        start_round = 0
        start_sample = 0
        print(f"Loaded {len(data)} samples | Bắt đầu từ Round 1, Sample 0")

    todo_ids = [i for i in range(len(data)) if data_dict[i]["text"]]

    # =========================
    # INFERENCE
    # =========================
    for r in range(start_round, num_rounds):
        col = f"paraphrase_r{r+1}"

        if r == start_round:
            ids_this_round = [i for i in todo_ids if i >= start_sample]
        else:
            ids_this_round = todo_ids

        skipped = len(todo_ids) - len(ids_this_round)
        print(f"\n[Round {r+1}/{num_rounds}] | Skip {skipped} samples đã có")

        for i in tqdm(ids_this_round, desc=f"Round {r+1}", leave=True):
            try:
                current_input = data_dict[i].get(f"paraphrase_r{r}", data_dict[i]["text"])

                prompt = model.build_prompt(current_input)
                raw = model.generate(prompt)
                para = clean_generated_text(raw).replace("\n", " ").replace("\r", " ")

                data_dict[i][col] = para

            except Exception as e:
                print(f"\n[ERROR] Sample {i}, Round {r+1}: {e}")

        # Checkpoint
        df = pd.DataFrame(list(data_dict.values()))
        df.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig", quoting=1)
        print(f"  → Checkpoint round {r+1} saved to {output_path}")

    print(f"\nDone inference. Saved → {output_path}")

    # =========================
    # EVALUATION
    # =========================
    print("\nRunning evaluation (all rounds)...")

    df = pd.read_csv(output_path, sep=";")

    evaluator = Evaluator(
        tokenizer_backend=cfg.get("evaluation", {}).get("tokenizer", "underthesea")
    )

    base_path = output_path.replace(".csv", "")
    summaries = []

    for r in tqdm(range(num_rounds), desc="Evaluation"):
        col = f"paraphrase_r{r+1}"
        if col not in df.columns:
            print(f"[WARN] Column {col} không tồn tại, bỏ qua.")
            continue

        df_round = df[["id", "text", col]].rename(columns={col: "paraphrase"}).copy()
        df_round = evaluator.evaluate_dataframe(df_round)

        summary = evaluator.summarize(df_round)
        summary["round"] = r + 1
        summaries.append(summary)

        df_round.to_csv(f"{base_path}_round{r+1}_detail.csv", index=False, sep=";")

    summary_df = pd.DataFrame(summaries)[
        ["round", "samples", "meteor_mean", "bertscore_mean",
         "ngram_mean", "editdistance_mean", "edit_similarity_mean"]
    ]
    summary_df.to_csv(f"{base_path}_summary.csv", index=False, sep=";")

    print("\n===== SUMMARY =====")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()