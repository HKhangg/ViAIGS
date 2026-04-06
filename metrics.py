from nltk.translate import meteor
from polyglot.text import Text, Word
from evaluate import load
bertscore = load("bertscore")
import ngram
import editdistance
import mauve
from mauve.compute_mauve import compute_mauve
from tqdm import tqdm

def custom_tokenizer(text, language='vi'):
    return list(Text(text, hint_language_code=language).words)

def get_meteor(df):
    metric = [""] * len(df)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        original = row.text
        obfuscated = row.generated
        try:
            metric[index] = round(meteor([custom_tokenizer(original)], custom_tokenizer(obfuscated)), 4)
        except:
            metric[index] = 0.0
    return metric

def get_bertscore(df):
    metric = [""] * len(df)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        original = row.text
        obfuscated = row.generated
        try:
            results  = bertscore.compute(predictions=[obfuscated], references=[original], model_type="bert-base-multilingual-cased")
            metric[index] = sum(results['f1']) / len(results['f1'])
        except:
            metric[index] = 0.0
    return metric

def get_ngram(df):
    metric = [""] * len(df)
    for index, row in tqdm(df.iterrows(), df.shape[0]):
        original = row.text
        obfuscated = row.generated
        try:
            metric[index] = round(ngram.NGram.compare(original, obfuscated, N=n), 4)
        except:
            metric[index] = 0.0
    return metric

def get_editdistance(df):
    metric = [""] * len(df)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        original = row.text
        obfuscated = row.generated
        try:
            metric[index] = editdistance.eval(original, obfuscated)
        except:
            metric[index] = 0.0
    return metric

