import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np

# class dataset
class ViAIGSDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        self.texts = data['text'].to_list()
        self.labels = data['label'].to_list()

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer(text, max_length = 512, padding = True, truncation = True, return_tensors = 'pt', return_attention_mask=True)
        # return input_ids, attention_mask and label
        return {"input_ids": encoding['input_ids'].squeeze(0), "attention_mask": encoding['attention_mask'].squeeze(0),"labels": torch.tensor(label)}

# class model
class AIDetection:
    def __init__(self, model, num_labels=2):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=num_labels
        )
        self.metric = evaluate.load("f1")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)
    
    def train(self, train, val, train_args):
        trainer = Trainer(
            model = self.model,
            args = train_args,
            train_dataset=train,
            eval_dataset=val,
            compute_metrics = self.compute_metrics,
        )
        return trainer.train()
    
train_dataset = "/kaggle/input/datasets/huykhang0106/viaigs-full-data/ViAIGS_dev.csv"
val_dataset = "/kaggle/input/datasets/huykhang0106/viaigs-full-data/ViAIGS_dev.csv"

args = TrainingArguments(output_dir='./test', num_train_epochs=3)

pipeline = AIDetection("microsoft/mdeberta-v3-base")
pipeline.train(train_dataset,val_dataset, args)

# metric
# training loop