# models/aya_model.py

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from .base_model import BaseLLM
from builders.registry import register_model
from utils.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from utils.postprocess import clean_generated_text


@register_model("aya")
class AyaLLM(BaseLLM):

    def __init__(self, config):
        super().__init__(config)
        self.load_model()

    def load_model(self):
        m_cfg = self.config["model"]

        # ====== 4bit quantization (giống Gemma) ======
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
            trust_remote_code=True
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            m_cfg["pretrained"],
            quantization_config=quant_cfg,
            torch_dtype=torch.bfloat16 if quant_cfg is None else None,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()

        # ====== EOS / PAD handling ======
        self._eos_token_id = self.model.config.eos_token_id
        self._pad_token_id = (
            self.model.config.pad_token_id
            or self.tokenizer.eos_token_id
        )

    def build_prompt(self, text: str):
        user_content = USER_PROMPT_TEMPLATE.format(text=text)

        # Aya = Seq2Seq → plain prompt OK
        return f"{SYSTEM_PROMPT}\n\n{user_content}"

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
                min_new_tokens=gen_cfg.get("min_new_tokens", 5),
                max_new_tokens=gen_cfg.get("max_new_tokens", 256),
                do_sample=gen_cfg.get("do_sample", True),
                temperature=gen_cfg.get("temperature", 1.0),
                top_p=gen_cfg.get("top_p", 0.95),
                top_k=gen_cfg.get("top_k", 50),
                eos_token_id=self._eos_token_id,
                pad_token_id=self._pad_token_id,
            )

        result = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return clean_generated_text(result)