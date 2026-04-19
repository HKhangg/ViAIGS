# utils/postprocess.py

def clean_generated_text(text: str) -> str:
    text = text.strip()
    for marker in ["USER:", "ASSISTANT:"]:
        if marker in text:
            text = text.split(marker)[0]
    return text.split("\n\n")[0].strip()