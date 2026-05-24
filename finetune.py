import argparse
import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from scipy.special import softmax
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # dùng để chạy trên 1 gpu
set_seed(42)
hf_token = os.getenv("HF_TOKEN", "")

target_map = {
            "microsoft/mdeberta-v3-base": ["query_proj", "key_proj", "value_proj"],
            "FacebookAI/xlm-roberta-large": [
                "query",
                "key",
                "value",
            ],
            "meta-llama/Meta-Llama-3-8B": [
                "q_proj",
                "k_proj",
                "v_proj",
            ],
            "mistralai/Mistral-7B-v0.3": [
                "q_proj",
                "k_proj",
                "v_proj",
            ],
        }

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

# metric
def macro_f1_at_fpr(y_true, y_score, target_fpr=0.05):
    thresholds = np.unique(y_score)
    best_f1 = 0.0
    best_threshold = 0.5
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        if fpr <= target_fpr:
            f1_macro = f1_score(y_true, y_pred, average="macro")
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_threshold = thr

    return best_f1, best_threshold


def compute_metrics(p: EvalPrediction):
    labels = p.label_ids
    logits = p.predictions

    probs = softmax(logits, axis=1)
    ai_probs = probs[:,1]
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    auc_roc = roc_auc_score(labels, ai_probs)
    pr_auc = average_precision_score(labels, ai_probs)

    macro_f1_5fpr, best_thr = macro_f1_at_fpr(labels, ai_probs, target_fpr=0.05)
    return {
        "accuracy": acc,
        "f1": f1,
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "macro_f1_5fpr": macro_f1_5fpr,
        "best_threshold_5fpr": best_thr,
    }

def load_model(model_name, use_peft, tokenizer):
    if use_peft:
        print("Xử dụng QLora")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16) #torch.float16

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,device_map="auto", quantization_config=quantization_config, cache_dir="./cache/", token=hf_token or None)
        model.config.pad_token_id = tokenizer.pad_token_id
        #lora
        model = prepare_model_for_kbit_training(model)  
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            target_modules=target_map.get(model_name, None),
            lora_dropout=0.1,
        )
        model = get_peft_model(model,peft_config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,device_map="auto", cache_dir="./cache/", token=hf_token or None) #torch_dtype=torch.float32
        model.config.pad_token_id = tokenizer.pad_token_id
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            target_modules=target_map.get(model_name, None),
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        # model = model.float()

    model.print_trainable_parameters()
    return model

def run_train(args):
    train_df = pd.read_csv(args.train_data)
    dev_df = pd.read_csv(args.dev_data)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = load_model(args.model_name, args.use_peft, tokenizer)

    train_dataset = ViAIGSDataset(train_df, tokenizer)
    dev_dataset = ViAIGSDataset(dev_df, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        logging_steps=10,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True, # Bật cái này giảm VRAM nhưng tăng thời gian train
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.1,
        bf16=True,
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
    trainer.save_metrics("eval", eval_metrics)

    print("Saving best adapter model")
    trainer.save_model("./best_adapter_model")
    tokenizer.save_pretrained("./best_adapter_model")

def load_model_from_checkpoint(model_name, checkpoint_path, tokenizer, use_peft=False):
    print(f"load checkpoint from: {checkpoint_path}")
    if use_peft:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, cache_dir="./cache/", token=hf_token or None, quantization_config=quantization_config, device_map="auto"
        )
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, cache_dir="./cache/", token=hf_token or None, torch_dtype=torch.bfloat16, device_map="auto"
        )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    return model

def run_test(args):
    test_df = pd.read_csv(args.test_data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    test_dataset = ViAIGSDataset(test_df, tokenizer)

    model = load_model_from_checkpoint(args.model_name, args.checkpoint, tokenizer)

    eval_args = TrainingArguments(
        output_dir='./results_test',
        per_device_eval_batch_size=32,
        report_to="none",
        remove_unused_columns=False,
        bf16=True,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
    )
    print("Start testing")
    raw_output = trainer.predict(test_dataset, metric_key_prefix="test")

    test_metrics = raw_output.metrics
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    print("Saving predictions")
    probs = softmax(raw_output.predictions, axis=1)

    best_thr = args.dev_threshold 

    result_df = test_df.copy()
    result_df["prob_human"] = probs[:, 0]
    result_df["prob_ai"]    = probs[:, 1]
    result_df["pred_label"] = np.argmax(raw_output.predictions, axis=1)
    result_df["pred_label_5fpr"] = (probs[:, 1] >= best_thr).astype(int)

    out_path = "predictions.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Predictions saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    #train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("train_data", type=str)
    train_parser.add_argument("dev_data", type=str)
    train_parser.add_argument("model_name", type=str)
    train_parser.add_argument("--use_peft", action="store_true")

    #test
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("test_data", type=str)
    test_parser.add_argument("model_name", type=str)
    test_parser.add_argument("checkpoint", type=str)
    test_parser.add_argument("--use_peft", action="store_true")
    test_parser.add_argument("--dev_threshold", type=float)

    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'test':
        run_test(args)