#!/usr/bin/env python3
"""
Check OpenAI batch job status and retrieve results, then export to CSV.
Usage: python check_batch_status.py <batch_id> <model_name_or_path> <original_dataset_path>
"""

import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if len(sys.argv) < 4:
    print("Usage: python check_batch_status.py <batch_id> <model_name_or_path> <dataset_path>")
    sys.exit(1)

batch_id = sys.argv[1]
MODEL = sys.argv[2]
DATASET = sys.argv[3]

model_name = MODEL.split('/')[-1]
def resolve_output_dir() -> str:
    env_output_dir = os.getenv("OUTPUT_DIR", "").strip()
    if env_output_dir: return env_output_dir
    if os.path.isdir("/kaggle/working"): return "/kaggle/working"
    if os.path.isdir("/content"): return "/content"
    return os.getcwd()

OUTPUT_DIR = resolve_output_dir()
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{model_name}.csv")
PARAPHRASED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{model_name}_paraphrased.csv")

def clean_generated_text(text: str) -> str:
    text = text.strip()
    for marker in ["USER:", "ASSISTANT:"]:
        if marker in text:
            text = text.split(marker)[0]
    return text.split("\n\n")[0].strip()

openai_api_key = os.getenv("OPENAI_API_KEY", "")
openai_organization = os.getenv("OPENAI_ORGANIZATION", "")
client = OpenAI(api_key=openai_api_key, organization=openai_organization)

print(f"[INFO] Checking status for Batch: {batch_id}")
batch_job = client.batches.retrieve(batch_id)
print(f"[INFO] Status: {batch_job.status} ({batch_job.request_counts.completed}/{batch_job.request_counts.total})")

if batch_job.status == "completed":
    print("\n[INFO] Batch completed! Processing results...")
    
  
    results_content = client.files.content(batch_job.output_file_id)
    results_by_id = {}
    
    for line in results_content.iter_lines():
        if not line: continue
        data = json.loads(line)
        custom_id = data.get("custom_id")
        
        if data.get("error"):
            results_by_id[custom_id] = ""
        else:
            body = data.get("response", {}).get("body", {})
            content = body.get("output_text", "")
            if not content and "output" in body:
                items = body.get("output", [])
                if items:
                    content = items[0].get("content", [{}])[0].get("text", "")
            
            results_by_id[custom_id] = clean_generated_text(content)

    print(f"[INFO] Loading original dataset: {DATASET}")
    df = pd.read_csv(DATASET, lineterminator='\n', escapechar='\\')
    df.columns = df.columns.str.strip()
    
    generated = [''] * len(df)
    for idx in range(len(df)):
        generated[idx] = results_by_id.get(f"request-{idx}", "")

    df['generated'] = generated
    df.to_csv(OUTPUT_PATH, index=False, escapechar='\\')
    print(f"[SUCCESS] Saved main output to: {OUTPUT_PATH}")

    df['text'] = df['generated']
    df['generated'] = ""
    df.to_csv(PARAPHRASED_OUTPUT_PATH, index=False, escapechar='\\')
    print(f"[SUCCESS] Saved paraphrased output to: {PARAPHRASED_OUTPUT_PATH}")

elif batch_job.status in ["failed", "expired", "cancelled"]:
    print(f"[ERROR] Batch job ended with status: {batch_job.status}")
else:
    print(f"[WAIT] Batch is still {batch_job.status}. Please try again later.")