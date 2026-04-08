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
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
model_name = MODEL.split('/')[-1]
model_name_lower = model_name.lower()
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

is_gemma = 'gemma' in model_name_lower
is_phogpt = 'phogpt' in model_name_lower
is_phogpt_chat = is_phogpt and 'chat' in model_name_lower
phogpt_revision = os.getenv("PHOGPT_REVISION", "main")

tokenizer = None
model = None

if text2text:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=CACHE, token=hf_token or None)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="auto", quantization_config=quantization_config, trust_remote_code=True, cache_dir=CACHE, token=hf_token or None)
elif is_phogpt:
    config = AutoConfig.from_pretrained(
        MODEL,
        trust_remote_code=True,
        cache_dir=CACHE,
        token=hf_token or None,
        revision=phogpt_revision,
    )
    if hasattr(config, "init_device") and torch.cuda.is_available():
        config.init_device = "cuda"
    if hasattr(config, "attn_config") and isinstance(config.attn_config, dict):
        config.attn_config["attn_impl"] = "torch"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
        cache_dir=CACHE,
        token=hf_token or None,
        revision=phogpt_revision,
    )

    phogpt_model_kwargs = {
        "config": config,
        "trust_remote_code": True,
        "cache_dir": CACHE,
        "token": hf_token or None,
        "revision": phogpt_revision,
        "device_map": "auto",
    }
    if use8bit:
        phogpt_model_kwargs["load_in_8bit"] = True
    else:
        phogpt_model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(MODEL, **phogpt_model_kwargs)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=CACHE, token=hf_token or None)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", quantization_config=quantization_config, trust_remote_code=True, cache_dir=CACHE, token=hf_token or None)

if model is not None:
    model = model.eval()

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

eos_token_ids = []
if tokenizer.eos_token_id is not None:
    eos_token_ids.append(tokenizer.eos_token_id)
eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
if isinstance(eot_token_id, int) and eot_token_id >= 0 and eot_token_id not in eos_token_ids:
    eos_token_ids.append(eot_token_id)

#generating texts
generated = [''] * len(df)

system_prompt = (
    "You are a Vietnamese text rewriting assistant. "
    "Rewrite the input social media text in Vietnamese using different words and sentence structure. "
    "Preserve the original meaning, tone, and key details. "
    "Return only the rewritten text. Do not explain."
)

phogpt_instruction = (
    "Hãy viết lại đoạn văn bản mạng xã hội sau bằng tiếng Việt, dùng từ ngữ và cấu trúc câu khác "
    "nhưng vẫn giữ nguyên ý nghĩa, giọng điệu và các chi tiết quan trọng. "
    "Chỉ trả về đoạn văn đã viết lại, không giải thích thêm."
)

with torch.no_grad():
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row.text
        user_prompt = (
            "Rewrite this Vietnamese social media text.\n\n"
            f"Input:\n{text}\n\n"
            "Output:"
        )

        uses_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template and not is_phogpt

        if uses_chat_template:
            if is_gemma:
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
            if is_phogpt_chat:
                fallback_prompt = (
                    f"### Câu hỏi: {phogpt_instruction}\n\n"
                    f"Văn bản gốc:\n{text}\n\n"
                    "### Trả lời:"
                )
            elif is_phogpt:
                fallback_prompt = (
                    f"### Câu hỏi: {phogpt_instruction}\n\n"
                    f"Văn bản gốc:\n{text}\n\n"
                    "### Trả lời:"
                )
            else:
                fallback_prompt = f'You are a helpful assistant.\n\nTask: Generate the text in Vietnamese similar to the input social media text but using different words and sentence composition.\n\nInput: {text}\n\nOutput:'
            inputs = tokenizer(fallback_prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
            print("chat_template is not apply")
        generation_kwargs = {
            "min_new_tokens": 5,
            "max_new_tokens": 200,
            "num_return_sequences": 1,
            "do_sample": True,
            "num_beams": 1,
            "top_k": 50,
            "top_p": 0.95,
        }
        if eos_token_ids:
            generation_kwargs["eos_token_id"] = eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0]
        if tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

        generated_ids = model.generate(**inputs, **generation_kwargs)
        
        if text2text:
            result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            prompt_length = inputs['input_ids'].shape[1]
            new_tokens = generated_ids[0][prompt_length:]
            result = tokenizer.decode(new_tokens, skip_special_tokens=True)
            if is_phogpt and "### Trả lời:" in result:
                result = result.split("### Trả lời:", 1)[1]
            result = result.replace("<|eot_id|>", "").strip()
            result = result.split('\n\n')[0].strip()
        
        generated[index] = result

print(generated)