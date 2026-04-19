# data/loader.py
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    return df