import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .base_model import BaseLLM
from builders.registry import register_model
from utils.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from utils.postprocess import clean_generated_text


@register_model("aya")
class AyaLLM(BaseLLM):

    def load_model(self):
        m_cfg = self.config["model"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            m_cfg["pretrained"],
            trust_remote_code=True
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            m_cfg["pretrained"],
            device_map="auto",
            torch_dtype=torch.bfloat16,   # hợp với aya
            trust_remote_code=True
        )

        self.model.eval()

    def build_prompt(self, text: str):
        # Aya = text2text → không cần role/chat
        user_content = USER_PROMPT_TEMPLATE.format(text=text)

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{user_content}"
        )

        return prompt

    def generate(self, prompt: str):
        gen_cfg = self.config["generation"]

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                min_new_tokens=gen_cfg.get("min_new_tokens", 5),
                max_new_tokens=gen_cfg.get("max_new_tokens", 256),
                do_sample=gen_cfg.get("do_sample", True),
                temperature=gen_cfg.get("temperature", 1.0),
                top_p=gen_cfg.get("top_p", 0.95),
                top_k=gen_cfg.get("top_k", 50),
            )

        result = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return clean_generated_text(result)