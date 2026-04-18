import os

from dotenv import load_dotenv

load_dotenv()

openai_organization = os.getenv("OPENAI_ORGANIZATION", "")
openai_api_key = os.getenv("OPENAI_API_KEY", "")
hf_token = os.getenv("HF_TOKEN", "")
CACHE = os.getenv("CACHE_DIR", "./cache/")

import sys
MODEL = sys.argv[1]
DATASET = sys.argv[2]


import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helper
model_name = MODEL.split('/')[-1]
model_name_lower = model_name.lower()

is_vicuna = "vicuna" in model_name_lower

def build_vicuna_prompt(system_message, user_message):
    return(
        f"{system_message.strip()} "
        f"USER: {user_message.strip()} "
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
df = df[:10]

df['text'] = df['text'].astype(str)
if "generated" in df.columns:
    df['generated'] = df['generated'].astype(str)
df.fillna("", inplace=True)


df['text'] = ["nan" if x=="" else x for x in df["text"]]
if 'generated' in df.columns:
    df['generated'] = ["nan" if x=="" else x for x in df["generated"]]

#load model
use4bit = True
use8bit = False
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True, load_in_4bit=use4bit, load_in_8bit=use8bit, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

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

eos_token_id = model.generation_config.eos_token_id
pad_token_id = model.generation_config.pad_token_id
if pad_token_id is None:
    pad_token_id = tokenizer.eos_token_id

if model is not None:
    model = model.eval()

#generating texts
generated = [''] * len(df)

system_prompt = (
    "You are a Vietnamese text rewriting assistant. "
    "Rewrite the input social media text in Vietnamese using different words and sentence structure. "
    "Preserve the original meaning, tone, and key details. "
    "Return only the rewritten text. Do not explain."
)

with torch.no_grad():
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row.text
        user_prompt = (
            "Rewrite this Vietnamese social media text.\n\n"
            f"Input:\n{text}\n\n"
            "Output:"
        )
        if is_vicuna:
            vicuna_prompt = build_vicuna_prompt(system_prompt, user_prompt)
            inputs = tokenizer(
                vicuna_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            print("vicuna prompt is applied")

        elif hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            if "gemma" in model_name.lower():
                gemma_user_prompt = (
                    f"{system_prompt}\n\n"
                    f"{user_prompt}"
                )
                messages = [
                    {"role": "user", "content": gemma_user_prompt},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", truncation=True, max_length=512).to(device)
            print("chat_template is apply")
        else:
            fallback_prompt = (
                f"{system_prompt}\n\n"
                f"{user_prompt}"
            )
            inputs = tokenizer(fallback_prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
            print("chat_template is not apply")
        generated_ids = model.generate(**inputs, min_new_tokens=5, max_new_tokens=200, num_return_sequences=1, do_sample=True, num_beams=1, top_k=50, top_p=0.95, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
        
        if text2text:
            result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            prompt_length = inputs['input_ids'].shape[1]
            new_tokens = generated_ids[0][prompt_length:]
            result = tokenizer.decode(new_tokens, skip_special_tokens=True)
            result = clean_generated_text(result)
        
        generated[index] = result

print(generated)