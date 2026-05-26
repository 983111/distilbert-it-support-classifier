"""
evaluate.py
Run held-out test evaluation for a fine-tuned DistilBERT IT support classifier.
Usage:
  python src/evaluate.py
  python src/evaluate.py --model_path outputs/distilbert-it-support --data_path data/support_tickets.csv
"""
import argparse
import csv
import random
from typing import List, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import pipeline
LABEL_NAMES = ["billing", "hardware", "network", "account", "software"]
SEED = 42
def load_test_split(data_path: str) -> Tuple[List[str], List[int]]:
    rows = []
    with open(data_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"text": row["text"], "label": int(row["label"])})
    random.seed(SEED)
    random.shuffle(rows)
    n = len(rows)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)
    test_rows = rows[n_train + n_val :]
    return [r["text"] for r in test_rows], [r["label"] for r in test_rows]
def evaluate(model_path: str, data_path: str) -> None:
    texts, y_true = load_test_split(data_path)
    clf = pipeline("text-classification", model=model_path, device=-1)
    outputs = clf(texts, truncation=True, max_length=128)
    y_pred = [LABEL_NAMES.index(o["label"]) for o in outputs]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    print("Evaluation on held-out test split")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nDetailed report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/distilbert-it-support")
    parser.add_argument("--data_path", type=str, default="data/support_tickets.csv")
    args = parser.parse_args()
    evaluate(args.model_path, args.data_path)
