# DistilBERT IT Support Ticket Classifier

Fine-tuned `distilbert-base-uncased` on a custom 300-row IT support ticket dataset for 5-class text classification, achieving **86.67% test accuracy**. Trained using the HuggingFace Trainer API and published to the HuggingFace Hub.

🤗 **Model on Hub:** [vishwajeet456/distilbert-it-support-classifier](https://huggingface.co/vishwajeet456/distilbert-it-support-classifier)

---

## Task

Automatic routing of IT support tickets into one of five operational categories, enabling help-desk teams to triage requests without manual reading.

| ID | Label | Example ticket |
|---|---|---|
| 0 | `billing` | "I was charged twice for my subscription this month." |
| 1 | `hardware` | "My laptop fan is making a loud grinding noise." |
| 2 | `network` | "I cannot connect to the VPN from home." |
| 3 | `account` | "My account was locked after too many failed login attempts." |
| 4 | `software` | "The application crashes every time I try to export to PDF." |

---

## Results

Trained for 5 epochs on CPU in under 2 minutes.

| Metric | Score |
|---|---|
| **Test Accuracy** | **86.67%** |
| **Test F1 (macro)** | **86.43%** |

Per-class breakdown on 30-sample held-out test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| billing | 0.75 | 1.00 | 0.86 | 6 |
| hardware | 1.00 | 1.00 | 1.00 | 7 |
| network | 0.60 | 1.00 | 0.75 | 3 |
| account | 1.00 | 0.56 | 0.71 | 9 |
| software | 1.00 | 1.00 | 1.00 | 5 |
| **avg** | **0.87** | **0.91** | **0.86** | **30** |

Validation accuracy reached **96.67%** by epoch 5.

---

## Dataset

300 synthetic IT support ticket samples, balanced across 5 classes (60 per class). Generated with controlled text augmentation (prefix/suffix variation) to simulate realistic help-desk phrasing.

**Split:** 80% train (240) · 10% validation (30) · 10% test (30)

---

## Training Setup

```python
TrainingArguments(
    num_train_epochs        = 5,
    per_device_train_batch_size = 16,
    learning_rate           = 2e-5,
    weight_decay            = 0.01,
    warmup_steps            = 24,
    lr_scheduler_type       = "cosine",
    eval_strategy           = "epoch",
    save_strategy           = "no",
)
```

- Base model: `distilbert-base-uncased`
- Framework: HuggingFace `Trainer` API
- Collation: `DataCollatorWithPadding` (dynamic padding)
- Hardware: CPU (GitHub Codespaces)
- Training time: ~2 minutes

---

## Project Structure

```
.
├── src/
│   ├── create_dataset.py        # generates data/support_tickets.csv (300 rows)
│   ├── train.py                 # HuggingFace Trainer fine-tuning pipeline
│   └── inference.py             # pipeline-based inference demo
├── data/
│   └── support_tickets.csv      # auto-created by create_dataset.py
├── DistilBERT_IT_Support_Classifier.ipynb   # self-contained Colab notebook
├── README.md                    # HuggingFace model card
├── GITHUB_README.md             # this file
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/983111/distilbert-it-support-classifier
cd distilbert-it-support-classifier
pip install -r requirements.txt

# Generate dataset
python src/create_dataset.py

# Train
python src/train.py

# Train + push to Hub
python src/train.py --push_to_hub --hub_model_id vishwajeet456/distilbert-it-support-classifier
```

---

## Inference

```python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="vishwajeet456/distilbert-it-support-classifier",
)

tickets = [
    "My laptop screen has a dead pixel cluster.",
    "I was charged twice this month.",
    "VPN keeps disconnecting every 15 minutes.",
    "My account got locked and the reset email never arrived.",
    "The app crashes every time I export to PDF.",
]

for ticket in tickets:
    result = clf(ticket, truncation=True, max_length=128)[0]
    print(f"[{result['label']:10s}] {result['score']:.1%}  {ticket}")
```

---

## Requirements

```
torch>=2.1.0
transformers>=4.40.0
datasets>=2.18.0
accelerate>=0.28.0
scikit-learn>=1.4.0
huggingface_hub>=0.22.0
numpy>=1.26.0
```

---

## CV Line

> Fine-tuned DistilBERT on custom 300-row IT support ticket dataset for 5-class text classification, achieving 86.7% accuracy. Published on HuggingFace Hub: huggingface.co/vishwajeet456/distilbert-it-support-classifier

---

## License

Apache 2.0
