import argparse
import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    EvalPrediction,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # dùng để chạy trên 1 gpu
set_seed(42)
hf_token = os.getenv("HF_TOKEN", "")

# class dataset
class ViAIGSDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.encoding = tokenizer(
            df['text'].astype(str).to_list(),
            add_special_tokens = True,
            max_length = max_length,
            padding = 'max_length',
            return_tensors = 'pt',
            truncation = True,
        )
        self.labels =  df['label'].astype(int).to_list()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx].detach().clone() for key, val in self.encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}

# metric
# training loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data", type=str)
    parser.add_argument("dev_data", type=str)
    parser.add_argument("model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--use_perf", action="store_true")
    # parser.add_argument("output_dir", type=str, default="./results")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_data)
    dev_df = pd.read_csv(args.dev_data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    quantization_config = None
    if args.use_perf:
        print("Xử dụng QLora")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16) #torch.float16

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2,device_map="auto", quantization_config=quantization_config, cache_dir="./cache/", token=hf_token or None)
        #lora
        model = prepare_model_for_kbit_training(model)  
        pert_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            target_modules=["query_proj", "key_proj", "value_proj"],
            lora_dropout=0.1,
        )
        model = get_peft_model(model,pert_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2,device_map="auto", quantization_config=quantization_config, cache_dir="./cache/", token=hf_token or None, torch_dtype=torch.float32)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            target_modules=["query_proj", "key_proj", "value_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model = model.float()
        model.print_trainable_parameters()

    train_dataset = ViAIGSDataset(train_df, tokenizer)
    dev_dataset = ViAIGSDataset(dev_df, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        logging_steps=10,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.1,
        optim="adamw_torch",
        # Thêm: clip gradient để tránh exploding gradient
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = dev_dataset,
        compute_metrics=compute_metrics
    )

    print("Start training")
    
    trainer.train()
    
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)