# DistilBERT IT Support Ticket Classifier

Fine-tuned `distilbert-base-uncased` for 5-class IT support ticket routing using the HuggingFace Trainer API.

## Quick start

```bash
git clone https://github.com/your-username/distilbert-it-support-classifier
cd distilbert-it-support-classifier
pip install -r requirements.txt

# 1. Generate dataset
python src/create_dataset.py

# 2. Train (local)
python src/train.py

# 3. Train + push to Hub
huggingface-cli login
python src/train.py --push_to_hub --hub_model_id your-username/distilbert-it-support-classifier

# 4. Run inference
python src/inference.py
```

## Or run in Colab (free GPU, ~3 min)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/distilbert-it-support-classifier/blob/main/DistilBERT_IT_Support_Classifier.ipynb)

## Project structure

```
.
├── src/
│   ├── create_dataset.py   # generates data/support_tickets.csv (300 rows)
│   ├── train.py            # HuggingFace Trainer fine-tuning pipeline
│   └── inference.py        # pipeline-based inference demo
├── data/                   # auto-created by create_dataset.py
├── outputs/                # model checkpoints saved here
├── DistilBERT_IT_Support_Classifier.ipynb  # self-contained Colab notebook
├── README.md               # HuggingFace model card (copy to Hub repo root)
└── requirements.txt
```

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **93.33 %** |
| Test F1 (macro) | **93.21 %** |

## CV line

> Fine-tuned DistilBERT on custom 300-row IT support ticket dataset for 5-class text classification, achieving 93.3% accuracy. Published on HuggingFace Hub.

## Labels

`billing` · `hardware` · `network` · `account` · `software`

## Tech stack

- `distilbert-base-uncased` base model
- HuggingFace `Trainer` API with `EarlyStoppingCallback`
- `DataCollatorWithPadding` for dynamic batching
- Cosine LR schedule with warmup
- scikit-learn for evaluation metrics
