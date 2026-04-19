# main.py

import argparse
import json
import yaml
import os
import pandas as pd

from builders.model_builder import build_model
from utils.postprocess import clean_generated_text
from evaluation.evaluator import Evaluator
from huggingface_hub import login


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    return parser.parse_args()


# =========================
# Resume helpers
# =========================
def load_done_ids(path):
    done_ids = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(int(json.loads(line)["id"]))
                except Exception:
                    continue
    return done_ids


def load_results(path):
    results = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except Exception:
                    continue
    return results


# =========================
# Main pipeline
# =========================
def main():
    args = parse_args()

    # ---- Load config ----
    with open(args.config_file, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # ---- HuggingFace login ----
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # ---- Build model ----
    model = build_model(cfg)

    # ---- Load dataset ----
    with open(cfg['data']['input'], encoding='utf-8') as f:
        data = json.load(f)

    output_path = cfg['data']['output']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ---- Resume ----
    done_ids = load_done_ids(output_path)
    print(f"Loaded {len(data)} samples | Skip {len(done_ids)} done samples")

    postprocess_cfg = cfg.get("postprocess", {})
    num_rounds = cfg.get("paraphrase", {}).get("num_rounds", 5)

    # =========================
    # Inference — chained paraphrase
    # =========================
    with open(output_path, 'a', encoding='utf-8') as f:
        for i, sample in enumerate(data):

            if i in done_ids:
                continue

            original_text = sample.get("text", "").strip()
            if not original_text:
                continue

            try:
                paraphrase_rounds = []
                current_input = original_text

                for round_idx in range(num_rounds):
                    prompt = model.build_prompt(current_input)
                    raw_output = model.generate(prompt)
                    paraphrase = clean_generated_text(raw_output)

                    paraphrase_rounds.append(paraphrase)
                    current_input = paraphrase  # output vòng này → input vòng sau

                    print(f"[{i+1}/{len(data)}] Round {round_idx+1}/{num_rounds} done")

                item = {
                    "id": i,
                    "text": original_text,
                    **{f"paraphrase_r{r+1}": paraphrase_rounds[r] for r in range(num_rounds)},
                }

                f.write(json.dumps(item, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"[ERROR] Sample {i}: {e}")

    # =========================
    # Evaluation — đánh giá từng vòng so với text gốc
    # =========================
    print("\nRunning evaluation...")

    results = load_results(output_path)

    if len(results) == 0:
        print("No results to evaluate.")
        return

    df = pd.DataFrame(results)

    evaluator = Evaluator(
        tokenizer_backend=cfg.get("evaluation", {}).get("tokenizer", "polyglot")
    )

    all_summaries = []
    base_path = output_path.replace(".json", "")

    for round_idx in range(num_rounds):
        col = f"paraphrase_r{round_idx+1}"

        if col not in df.columns:
            print(f"[WARN] Column {col} không tồn tại, bỏ qua.")
            continue

        # Tạo df tạm, đổi tên cột thành "paraphrase" để evaluator nhận đúng key
        df_round = df[["id", "text", col]].rename(columns={col: "paraphrase"}).copy()

        print(f"\n--- Evaluating Round {round_idx+1} ---")
        df_round = evaluator.evaluate_dataframe(df_round)

        summary = evaluator.summarize(df_round)
        summary["round"] = round_idx + 1
        all_summaries.append(summary)

        print(f"Round {round_idx+1}: {summary}")

        # Lưu detailed metrics riêng cho từng vòng
        df_round.to_csv(f"{base_path}_round{round_idx+1}_detailed.csv", index=False)

    # ---- Lưu summary tổng hợp tất cả các vòng ----
    summary_df = pd.DataFrame(all_summaries)[
        ["round", "samples", "meteor_mean", "bertscore_mean",
         "ngram_mean", "editdistance_mean", "edit_similarity_mean"]
    ]

    summary_path = f"{base_path}_rounds_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n===== SUMMARY ALL ROUNDS =====")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()