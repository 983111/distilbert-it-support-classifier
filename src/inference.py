"""
inference.py
────────────
Load the fine-tuned model (local or from Hub) and classify new tickets.

Usage:
  python inference.py                         # demo with built-in examples
  python inference.py --model_path outputs/distilbert-it-support
  python inference.py --hub your-username/distilbert-it-support-classifier
"""

import argparse
from transformers import pipeline


DEMO_TEXTS = [
    "My laptop screen has a dead pixel cluster in the top-right corner.",
    "I was charged twice this month and need an immediate refund.",
    "VPN keeps disconnecting every 15 minutes while working from home.",
    "My account got locked and the reset email never arrived.",
    "The application crashes every time I try to export a report to PDF.",
    "The printer on the second floor is showing offline status.",
    "My invoice shows I was billed at the old rate after the plan change.",
    "I cannot connect to any internal resources through the office network.",
]


def classify(texts: list[str], model_path: str) -> None:
    print(f"\nLoading model from: {model_path}\n")
    clf = pipeline(
        "text-classification",
        model=model_path,
        device=-1,  # CPU; set to 0 for GPU
    )

    print(f"{'─'*65}")
    print(f"{'TICKET TEXT':<45} {'LABEL':<12} {'SCORE':>6}")
    print(f"{'─'*65}")

    results = clf(texts, truncation=True, max_length=128)
    for text, result in zip(texts, results):
        label = result["label"]
        score = result["score"]
        display_text = (text[:42] + "...") if len(text) > 45 else text
        print(f"{display_text:<45} {label:<12} {score:>5.1%}")

    print(f"{'─'*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/distilbert-it-support",
                        help="Local path to saved model directory")
    parser.add_argument("--hub", type=str, default=None,
                        help="HuggingFace Hub model ID (overrides --model_path)")
    args = parser.parse_args()

    model_source = args.hub if args.hub else args.model_path
    classify(DEMO_TEXTS, model_source)
