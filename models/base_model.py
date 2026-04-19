# models/base_model.py
class BaseLLM:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        raise NotImplementedError

    def build_prompt(self, text: str) -> str:
        raise NotImplementedError

    def generate(self, prompt: str) -> str:
        raise NotImplementedError