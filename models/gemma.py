# models/gemma_model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base_model import BaseLLM
from builders.registry import register_model
from utils.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from utils.postprocess import clean_generated_text


@register_model("gemma")
class GemmaLLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)
        self.load_model()

    def load_model(self):
        m_cfg = self.config["model"]

        quant_cfg = None
        if m_cfg.get("load_in_4bit", False):
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            m_cfg["pretrained"],
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            m_cfg["pretrained"],
            quantization_config=quant_cfg,
            torch_dtype=torch.bfloat16 if quant_cfg is None else None,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self._eos_token_id = self.model.generation_config.eos_token_id
        self._pad_token_id = (
            self.model.generation_config.pad_token_id
            or self.tokenizer.eos_token_id
        )

    def build_prompt(self, text: str):
        user_content = USER_PROMPT_TEMPLATE.format(text=text)

        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_content}"}
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: str):
        gen_cfg = self.config["generation"]

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                min_new_tokens=gen_cfg["min_new_tokens"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=gen_cfg["do_sample"],
                temperature=gen_cfg.get("temperature", 1.0),
                top_p=gen_cfg.get("top_p", 0.95),
                top_k=gen_cfg.get("top_k", 50),
                eos_token_id=self._eos_token_id,     # FIX #4
                pad_token_id=self._pad_token_id,     # FIX #4
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]

        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return clean_generated_text(result)