"""
train.py
────────
Fine-tunes DistilBERT on the IT Support Ticket dataset using the
HuggingFace Trainer API, evaluates on a held-out test split, and
optionally pushes the final model + tokenizer to HuggingFace Hub.

Usage:
  python train.py                          # train locally
  python train.py --push_to_hub           # train + push to Hub
  python train.py --hub_model_id your_username/distilbert-it-support-classifier

Requirements:
  pip install transformers datasets scikit-learn accelerate torch huggingface_hub
"""

import argparse
import os
import csv
import random
import numpy as np
from pathlib import Path

# ── HuggingFace imports ──────────────────────────────────────────────────────
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "distilbert-base-uncased"
NUM_LABELS = 5
LABEL_NAMES = ["billing", "hardware", "network", "account", "software"]
DATA_PATH = "data/support_tickets.csv"
OUTPUT_DIR = "outputs/distilbert-it-support"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv_dataset(path: str) -> DatasetDict:
    """Load CSV, split 80/10/10 into train/val/test."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"text": row["text"], "label": int(row["label"])})

    random.shuffle(rows)
    n = len(rows)
    n_train = int(0.80 * n)
    n_val   = int(0.10 * n)

    splits = {
        "train": rows[:n_train],
        "validation": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }

    features = Features({
        "text": Value("string"),
        "label": ClassLabel(num_classes=NUM_LABELS, names=LABEL_NAMES),
    })

    return DatasetDict({
        split: Dataset.from_list(data, features=features)
        for split, data in splits.items()
    })


def tokenize_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )
    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    print(f"\n{'='*60}")
    print(f"  DistilBERT Fine-Tuning: IT Support Ticket Classifier")
    print(f"{'='*60}\n")

    # 1. Load & tokenize data
    print("▶ Loading dataset …")
    raw_datasets = load_csv_dataset(DATA_PATH)
    print(f"  Train: {len(raw_datasets['train'])} | "
          f"Val: {len(raw_datasets['validation'])} | "
          f"Test: {len(raw_datasets['test'])}")

    print("▶ Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    print("▶ Tokenizing …")
    tokenized = tokenize_dataset(raw_datasets, tokenizer)

    # 2. Load model
    print("▶ Loading model …")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=NUM_LABELS,
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        seed=SEED,
        report_to="none",        # set to "wandb" if you use W&B
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_strategy="end",
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 5. Train
    print("\n▶ Training …")
    trainer.train()

    # 6. Evaluate on held-out test set
    print("\n▶ Evaluating on test set …")
    test_preds_output = trainer.predict(tokenized["test"])
    test_preds = np.argmax(test_preds_output.predictions, axis=-1)
    test_labels = test_preds_output.label_ids

    accuracy = accuracy_score(test_labels, test_preds)
    f1_macro = f1_score(test_labels, test_preds, average="macro")

    print(f"\n  Test Accuracy : {accuracy*100:.2f}%")
    print(f"  Test F1 Macro : {f1_macro*100:.2f}%")
    print("\n  Per-class report:")
    print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES))

    # Save metrics to file
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{OUTPUT_DIR}/test_metrics.txt", "w") as mf:
        mf.write(f"Test Accuracy: {accuracy*100:.2f}%\n")
        mf.write(f"Test F1 Macro: {f1_macro*100:.2f}%\n\n")
        mf.write(classification_report(test_labels, test_preds, target_names=LABEL_NAMES))
    print(f"\n  Metrics saved to {OUTPUT_DIR}/test_metrics.txt")

    # 7. Push to Hub
    if args.push_to_hub:
        print(f"\n▶ Pushing model to HuggingFace Hub → {args.hub_model_id} …")
        trainer.push_to_hub(commit_message="Fine-tuned DistilBERT for IT support ticket classification")
        print("  Done! 🎉")

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Final Test Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*60}\n")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to HuggingFace Hub after training")
    parser.add_argument("--hub_model_id", type=str,
                        default="your-username/distilbert-it-support-classifier",
                        help="HuggingFace Hub model repo (username/model-name)")
    args = parser.parse_args()
    main(args)
