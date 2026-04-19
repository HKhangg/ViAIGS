import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseLLM
from builders.registry import register_model
from utils.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from utils.postprocess import clean_generated_text


def build_vicuna_prompt(system_message, user_message):
    return (
        f"{system_message.strip()} "
        f"USER: {user_message.strip()} "
        "ASSISTANT:"
    )


@register_model("vicuna")
class VicunaLLM(BaseLLM):

    def load_model(self):
        m_cfg = self.config["model"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            m_cfg["pretrained"]
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            m_cfg["pretrained"],
            torch_dtype=torch.float16,   # Vicuna dùng float16
            device_map="auto",
        )

        self.model.eval()

    def build_prompt(self, text: str):
        user_content = USER_PROMPT_TEMPLATE.format(text=text)

        prompt = build_vicuna_prompt(
            SYSTEM_PROMPT,
            user_content
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
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]

        result = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        return clean_generated_text(result)