openai_organization = ""
openai_api_key = ""
hf_token = ""
CACHE = "./cache/"

import sys
MODEL = sys.argv[1]
DATASET = sys.argv[2]


import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
model_name = MODEL.split('/')[-1]
df = pd.read_csv(DATASET, lineterminator='\n', escapechar='\\')
df = df[:10]
df['text'] = df['text'].astype(str)
if "generated" in df.columns:
    df['generated'] = df['generated'].astype(str)
df.fillna("", inplace=True)
df = df[df.label != 'label'].reset_index(drop=True)

df['text'] = ["nan" if x=="" else x for x in df["text"]]
if 'generated' in df.columns:
    df['generated'] = ["nan" if x=="" else x for x in df["generated"]]

#load model
use4bit = True
use8bit = False
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True, load_in_4bit=use4bit, load_in_8bit=use8bit, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, cache_dir=CACHE)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", quantization_config=quantization_config, trust_remote_code=True, cache_dir=CACHE)

if model is not None:
    model = model.eval()

#generating texts
generated = [""] * len(df)
with torch.no_grad():
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = row.text.replace("\n\n","\n")
        prompt = f'You are a helpful assistent.\n\nTask: Generate the text in Vietnamese similar to the input social media text but using different words and sentence composition.\n\nInput: {text}\n\nOutput:'

        input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).input_ids

        generated_ids = model.generate(input_ids=input_ids.to(device), min_new_tokens=5, max_new_tokens=200, num_return_sequences=1, do_sample=True, num_beams=1, top_k=50, top_p=0.95)
        result = tokenizer.decode(generated_ids, skip_special_tokens=True)

        generated[index] = result

print(generated)