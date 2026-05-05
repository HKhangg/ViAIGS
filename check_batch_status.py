#!/usr/bin/env python3
"""
Check OpenAI batch job status and retrieve results.
Usage: python check_batch_status.py <batch_id> [output_dir]
"""

import os
import sys
import json
import jsonlines
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python check_batch_status.py <batch_id> [output_dir]")
    sys.exit(1)

batch_id = sys.argv[1]
output_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()

openai_api_key = os.getenv("OPENAI_API_KEY", "")
openai_organization = os.getenv("OPENAI_ORGANIZATION", "")

if not openai_api_key:
    print("[ERROR] OPENAI_API_KEY not set in environment")
    sys.exit(1)

client = OpenAI(api_key=openai_api_key, organization=openai_organization)

print(f"[INFO] Checking batch status for {batch_id}...")
batch_job = client.batches.retrieve(batch_id)

print(f"[INFO] Status: {batch_job.status}")
print(f"[INFO] Created at: {batch_job.created_at}")
print(f"[INFO] Completed at: {batch_job.completed_at}")
print(f"[INFO] Request counts:")
print(f"  - Total: {batch_job.request_counts.total}")
print(f"  - Completed: {batch_job.request_counts.completed}")
print(f"  - Failed: {batch_job.request_counts.failed}")
print(f"  - Expired: {batch_job.request_counts.expired}")

if batch_job.status == "completed":
    print("\n[INFO] Batch completed! Retrieving results...")
    
    results_content = client.files.content(batch_job.output_file_id)
    
    results_by_id = {}
    errors = []
    
    for line in results_content.iter_lines():
        if not line:
            continue
            
        data = json.loads(line)
        custom_id = data.get("custom_id")
        
        if data.get("error"):
            error_msg = data.get("error")
            print(f"[ERROR] Request {custom_id} failed: {error_msg}")
            errors.append((custom_id, error_msg))
            results_by_id[custom_id] = ""
        else:
            body = data.get("response", {}).get("body", {})
            content = ""

            if "output_text" in body:
                content = body["output_text"]
            
            elif "output" in body:
                outputs = body.get("output", [])
                if outputs and isinstance(outputs, list):
                    item_contents = outputs[0].get("content", [])
                    for part in item_contents:
                        if part.get("type") in ["text", "output_text"]:
                            content = part.get("text", "")
                            break
            
            results_by_id[custom_id] = content.strip()
    
    print(f"[INFO] Retrieved {len(results_by_id)} successful results")
    if errors:
        print(f"[WARNING] {len(errors)} requests failed")
    
    # Save results to file
    results_file = os.path.join(output_dir, f"batch_{batch_id}_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_by_id, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Results saved to {results_file}")
    
    if errors:
        errors_file = os.path.join(output_dir, f"batch_{batch_id}_errors.json")
        with open(errors_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Errors saved to {errors_file}")

elif batch_job.status == "in_progress":
    print("\n[INFO] Batch is still processing...")
    print(f"[INFO] Please check again later")

elif batch_job.status in ["failed", "expired"]:
    print(f"\n[ERROR] Batch job {batch_job.status}!")
    if batch_job.errors:
        print(f"[ERROR] Details: {batch_job.errors}")

print(f"\n[INFO] To check again, run:")
print(f"      python check_batch_status.py {batch_id} {output_dir}")
