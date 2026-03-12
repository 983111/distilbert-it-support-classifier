---
language: en
license: apache-2.0
tags:
  - text-classification
  - distilbert
  - it-support
  - ticket-classification
  - nlp
datasets:
  - custom
metrics:
  - accuracy
  - f1
base_model: distilbert-base-uncased
pipeline_tag: text-classification
---

# distilbert-it-support-classifier

A fine-tuned **DistilBERT** model for automatic IT support ticket classification. Given a raw ticket body (a sentence or two of plain English), the model routes the ticket into one of five operational categories, enabling help-desk teams to triage requests without manual reading.

## Model Description

| Attribute | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Task | Multi-class text classification |
| Number of labels | 5 |
| Max sequence length | 128 tokens |
| Training framework | HuggingFace Trainer API |
| Hardware used | Single CPU / GPU |

## Labels

| ID | Label | Description |
|---|---|---|
| 0 | `billing` | Payment issues, invoices, refunds, subscription charges |
| 1 | `hardware` | Physical device problems — laptops, peripherals, printers |
| 2 | `network` | Connectivity, VPN, DNS, Wi-Fi, firewall issues |
| 3 | `account` | Login, password reset, MFA, provisioning, access control |
| 4 | `software` | Application crashes, updates, licences, driver issues |

## Dataset

The model was trained on a **custom synthetic dataset** of 300 IT support ticket samples (60 per class), constructed to reflect realistic help-desk language patterns. The dataset was designed to:

- Cover the five most common IT support categories in enterprise environments
- Include natural language variation through prefix/suffix augmentation
- Maintain balanced class distribution to avoid label bias

**Split:** 80 % train · 10 % validation · 10 % test (stratified by label)

> **Why synthetic?** Synthetic data lets anyone reproduce the experiment without privacy concerns around real ticket data, while still capturing the vocabulary and phrasing of genuine IT support requests. The pipeline is identical to what you would run on proprietary ticket data — swap the CSV and retrain.

## Training Procedure

```python
TrainingArguments(
    num_train_epochs        = 5,
    per_device_train_batch_size = 16,
    learning_rate           = 2e-5,
    weight_decay            = 0.01,
    warmup_ratio            = 0.1,
    lr_scheduler_type       = "cosine",
    eval_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "f1_macro",
)
```

Early stopping (patience = 2 epochs) was applied to prevent overfitting on the small dataset. The best checkpoint by validation F1-macro was retained.

## Evaluation Results

Evaluated on the held-out test split (30 samples):

| Metric | Score |
|---|---|
| **Test Accuracy** | **93.33 %** |
| **Test F1 (macro)** | **93.21 %** |

Per-class breakdown:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| billing | 1.00 | 0.83 | 0.91 | 6 |
| hardware | 0.86 | 1.00 | 0.92 | 6 |
| network | 1.00 | 1.00 | 1.00 | 6 |
| account | 1.00 | 0.83 | 0.91 | 6 |
| software | 0.86 | 1.00 | 0.92 | 6 |

> Results are averaged over 3 random seeds; numbers above are from seed 42.

## How to Use

### With the `pipeline` API (simplest)

```python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="your-username/distilbert-it-support-classifier",
)

tickets = [
    "My laptop fan is making a grinding noise and overheating.",
    "I was charged twice for my subscription this month.",
    "VPN keeps disconnecting every 10 minutes from home.",
]

for ticket in tickets:
    result = clf(ticket, truncation=True, max_length=128)
    print(f"[{result[0]['label']}]  {ticket[:60]}")
```

### With `AutoModelForSequenceClassification`

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "your-username/distilbert-it-support-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "The application crashes when I try to export a PDF."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_id = logits.argmax(-1).item()
print(model.config.id2label[predicted_id])  # → software
```

## Limitations

- Trained on synthetic data — real-world performance on domain-specific ticket systems may vary and warrants validation on internal data before production use.
- 128-token max length truncates very long ticket bodies; prepend the most important sentence for best results.
- English only; no multilingual support in this checkpoint.
- The five categories cover common enterprise IT scenarios but are not exhaustive — tickets outside this taxonomy will be misrouted.

## Intended Use

This model is intended for:
- Automated tier-1 triage in IT help-desk systems
- Priority-routing integrations (e.g. Jira, ServiceNow, Zendesk)
- Research and experimentation with lightweight text classifiers

It is **not** intended for:
- High-stakes automated decision-making without human review
- Medical, legal, or financial routing

## Why I Built This

This project demonstrates the end-to-end ML workflow of fine-tuning a pre-trained transformer for a practical classification task:

1. **Dataset construction** — writing a principled data generation script with label balance and text augmentation
2. **HuggingFace Trainer API** — using `TrainingArguments`, `Trainer`, `EarlyStoppingCallback`, and `DataCollatorWithPadding` rather than a manual training loop
3. **Evaluation rigour** — computing accuracy and macro-F1 on a held-out test split, plus a full per-class classification report
4. **Model publishing** — pushing tokenizer + model + model card to the Hub for reproducibility

The domain (IT support routing) is deliberately practical: every company with an internal help desk has this problem, making the deliverable immediately relatable in an applied-AI context.

## Training Code

Full training code and dataset generation scripts are available at:
[github.com/your-username/distilbert-it-support-classifier](https://github.com/your-username/distilbert-it-support-classifier)

## Citation

If you use this model or methodology, please cite:

```bibtex
@misc{distilbert-it-support-2024,
  author    = {Vishwajeet Adkine},
  title     = {DistilBERT Fine-tuned for IT Support Ticket Classification},
  year      = {2024},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/your-username/distilbert-it-support-classifier}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
