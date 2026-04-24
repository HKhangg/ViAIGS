import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

openai_organization = os.getenv("OPENAI_ORGANIZATION", "")
openai_api_key = os.getenv("OPENAI_API_KEY", "")
hf_token = os.getenv("HF_TOKEN", "")
together_api_key = os.getenv("TOGETHER_API_KEY", "")
CACHE = os.getenv("CACHE_DIR", "./cache/")

MODEL = sys.argv[1]
DATASET = sys.argv[2]
BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 2

import pandas as pd
import numpy as np
from tqdm import tqdm

# helper
model_name = MODEL.split('/')[-1]
model_name_lower = model_name.lower()

def resolve_output_dir() -> str:
    env_output_dir = os.getenv("OUTPUT_DIR", "").strip()
    if env_output_dir:
        return env_output_dir
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working"
    if os.path.isdir("/content"):
        return "/content"
    return os.getcwd()


OUTPUT_DIR = resolve_output_dir()
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{model_name}.csv")
PARAPHRASED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{model_name}_paraphrased.csv")

use_together_api = bool(together_api_key)

is_vicuna = "vicuna" in model_name_lower

def build_vicuna_prompt(system_message, user_message):
    return (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        f"USER: {system_message.strip()}\n\n{user_message.strip()}\n"
        "ASSISTANT:"
    )

def clean_generated_text(text: str) -> str:
    text = text.strip()
    for marker in ["USER:", "ASSISTANT:"]:
        if marker in text:
            text = text.split(marker)[0]
    return text.split("\n\n")[0].strip()

#load data
df = pd.read_csv(DATASET, lineterminator='\n', escapechar='\\')
df.columns = df.columns.str.strip()

df['text'] = df['text'].astype(str)
if "generated" in df.columns:
    df['generated'] = df['generated'].astype(str)
df.fillna("", inplace=True)


df['text'] = ["nan" if x=="" else x for x in df["text"]]
if 'generated' in df.columns:
    df['generated'] = ["nan" if x=="" else x for x in df["generated"]]

#generating texts
generated = [''] * len(df)

system_prompt = (
    "You are a Vietnamese text rewriting assistant. "
    "Rewrite the input social media text in Vietnamese using different words and sentence structure. "
    "Preserve the original meaning, tone, and key details. "
    "Return only the rewritten text. Do not explain."
)


def build_user_prompt(text):
    return (
        "Rewrite this Vietnamese social media text.\n\n"
        f"Input:\n{text}\n\n"
        "Output:"
    )


if use_together_api:
    # ── Together AI API mode ──
    from together import Together

    client = Together(api_key=together_api_key)
    print(f"[INFO] Using Together AI API with model: {MODEL}")

    def generate_via_api(text, max_retries=5):
        user_prompt = build_user_prompt(text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=200,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                )
                result = response.choices[0].message.content.strip()
                # truncate if result exceeds input length + 10 words
                input_words = text.split()
                result_words = result.split()
                if len(result_words) > len(input_words) + 10:
                    result = ' '.join(result_words[:len(input_words) + 10])
                return result
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    wait = 2 ** attempt
                    print(f"[WARN] Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[ERROR] API call failed: {e}")
                    if attempt == max_retries - 1:
                        return ""
                    time.sleep(1)
        return ""

    for i in tqdm(range(len(df)), desc="Generating via Together AI"):
        generated[i] = generate_via_api(df.iloc[i]["text"])

else:
    # ── Local model mode ──
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    import gc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load model
    use4bit = True
    use8bit = False
    quantization_config = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=True, 
        load_in_4bit=use4bit, 
        load_in_8bit=use8bit, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Memory optimization
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    text2text = False
    if 'aya-101' in model_name: text2text = True

    tokenizer = None
    model = None

    if text2text:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=CACHE, token=hf_token or None)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="auto", quantization_config=quantization_config, trust_remote_code=True, cache_dir=CACHE, token=hf_token or None)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=CACHE, token=hf_token or None)
        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", quantization_config=quantization_config, trust_remote_code=True, cache_dir=CACHE, token=hf_token or None)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For single sentence inference, use right padding for consistency
    if not text2text:
        tokenizer.padding_side = "right"  # Changed from "left" for single sentence inference

    eos_token_id = model.generation_config.eos_token_id
    pad_token_id = model.generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    if model is not None:
        model = model.eval()

    def build_batch_inputs(texts):
        user_prompts = [build_user_prompt(text) for text in texts]

        if is_vicuna:
            prompts = [build_vicuna_prompt(system_prompt, user_prompt) for user_prompt in user_prompts]
            return tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            if "gemma" in model_name.lower():
                messages = [
                    [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
                    for user_prompt in user_prompts
                ]
            else:
                messages = [
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    for user_prompt in user_prompts
                ]
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
            except Exception:
                # Fallback: simple prompt without system message
                prompts = [user_prompt for user_prompt in user_prompts]
                return tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

        # Default: simple prompts for single sentence inference
        prompts = [user_prompt for user_prompt in user_prompts]
        return tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)


    def truncate_to_input_length(result, text, extra_tokens=10):
        input_ids = tokenizer(text)['input_ids']
        result_ids = tokenizer(result)['input_ids']
        max_len = len(input_ids) + extra_tokens
        if len(result_ids) > max_len:
            result = tokenizer.decode(result_ids[:max_len], skip_special_tokens=True)
        return result

    def decode_batch_outputs(inputs, generated_ids, batch_texts):
        if text2text:
            decoded = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_ids]
            return [truncate_to_input_length(r, t) for r, t in zip(decoded, batch_texts)]

        input_length = inputs["input_ids"].shape[1]
        results = []
        for output_ids, orig_text in zip(generated_ids, batch_texts):
            new_tokens = output_ids[input_length:]
            result = tokenizer.decode(new_tokens, skip_special_tokens=True)
            result = clean_generated_text(result)
            result = truncate_to_input_length(result, orig_text)
            results.append(result)
        return results

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(df), BATCH_SIZE), total=(len(df) + BATCH_SIZE - 1) // BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(df))
            batch_texts = df.iloc[batch_start:batch_end]["text"].tolist()

            inputs = build_batch_inputs(batch_texts)
            generated_ids = model.generate(
                **inputs,
                min_new_tokens=5,
                max_new_tokens=200,
                num_return_sequences=1,
                do_sample=True,
                num_beams=1,
                top_k=50,
                top_p=0.95,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

            batch_results = decode_batch_outputs(inputs, generated_ids, batch_texts)
            generated[batch_start:batch_end] = batch_results
            
            # Clear GPU memory after each batch
            del inputs, generated_ids
            torch.cuda.empty_cache()
            gc.collect()

    # Clean up model after inference to free VRAM
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

df['generated'] = generated
df.to_csv(OUTPUT_PATH, index=False, escapechar='\\')

#modify to make it ready as the input to the next iteration of paraphrasing
df['text'] = df['generated']
df['generated'] = ""
df.to_csv(PARAPHRASED_OUTPUT_PATH, index=False, escapechar='\\')