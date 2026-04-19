import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base_model import BaseLLM
from builders.registry import register_model
from utils.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from utils.postprocess import clean_generated_text


def build_vicuna_prompt(system_message, user_message):
    return (
        f"{system_message.strip()}\n\n"
        f"USER: {user_message.strip()}\n"
        f"ASSISTANT:"
    )


@register_model("vicuna")
class VicunaLLM(BaseLLM):

    def __init__(self, config):
        super().__init__(config)
        self.load_model()

    def load_model(self):
        m_cfg = self.config["model"]

        # ===== 4bit support (GIỐNG GEMMA/MISTRAL) =====
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
            use_fast=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            m_cfg["pretrained"],
            quantization_config=quant_cfg,
            torch_dtype=torch.float16 if quant_cfg is None else None,
            device_map="auto",
        )

        self.model.eval()

        # ===== PAD FIX (critical for LLaMA/Vicuna) =====
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._eos_token_id = self.tokenizer.eos_token_id
        self._pad_token_id = self.tokenizer.pad_token_id

    def build_prompt(self, text: str):
        user_content = USER_PROMPT_TEMPLATE.format(text=text)

        return build_vicuna_prompt(
            SYSTEM_PROMPT,
            user_content
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
                min_new_tokens=gen_cfg.get("min_new_tokens", 5),
                max_new_tokens=gen_cfg.get("max_new_tokens", 256),
                do_sample=gen_cfg.get("do_sample", True),
                temperature=gen_cfg.get("temperature", 1.0),
                top_p=gen_cfg.get("top_p", 0.95),
                top_k=gen_cfg.get("top_k", 50),
                eos_token_id=self._eos_token_id,
                pad_token_id=self._pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]

        result = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        return clean_generated_text(result)