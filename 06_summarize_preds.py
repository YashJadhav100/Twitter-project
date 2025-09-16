# 06_summarize_preds.py
import argparse, pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV from 05_infer_twitter.py")
    ap.add_argument("--out_table", default="summary_table.csv")
    ap.add_argument("--out_plot", default="summary_plot.png")
    ap.add_argument("--out_examples", default="examples_by_label.txt")
    ap.add_argument("--per_label", type=int, default=2, help="Examples per label")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    # Basic frequency table
    counts = df["pred_label"].value_counts().sort_values(ascending=False)
    counts.to_csv(args.out_table, header=["count"])
    print("Saved table ->", args.out_table)

    # Bar chart (matplotlib, no custom colors)
    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title("Predicted Emotion Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=200)
    print("Saved plot  ->", args.out_plot)

    # Examples per label
    by_label = defaultdict(list)
    for _, row in df.iterrows():
        by_label[row["pred_label"]].append((row["id"], row["pred_score"], row["text"]))

    with open(args.out_examples, "w", encoding="utf-8") as f:
        for label in counts.index.tolist():
            f.write(f"\n=== {label} ===\n")
            samples = by_label[label]
            random.shuffle(samples)
            for (tid, score, text) in samples[:args.per_label]:
                f.write(f"- id={tid} | score={score:.3f}\n  {text}\n")
    print("Saved examples ->", args.out_examples)

if __name__ == "__main__":
    main()
