# 05_infer_twitter.py
import argparse, torch, pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax

from model_esa import EmotionAwareRoBERTa
from collator_tfidf import tfidf, tok  # reuse fitted TF-IDF + tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Get label names from GoEmotions (simplified)
try:
    ge = load_dataset("go_emotions", "simplified")
    LABELS = ge["train"].features["labels"].feature.names
except Exception:
    LABELS = [
        'admiration','amusement','anger','annoyance','approval','caring','confusion',
        'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
        'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
        'pride','realization','relief','remorse','sadness','surprise','neutral'
    ]

class TwitterSet(Dataset):
    def __init__(self, df, text_col="text", id_col="id", max_len=128):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.id_col = id_col
        self.max_len = max_len
        self.tokenizer = tok
        self.tfidf = tfidf
        self.vocab = tfidf.vocabulary_
        self.idf = tfidf.idf_
        self.specials = {tok.cls_token, tok.sep_token, tok.pad_token}

    def __len__(self):
        return len(self.df)

    def _tfidf_gate_for_ids(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        scores = []
        for t in tokens:
            if t in self.specials:
                scores.append(1.0)
                continue
            word = t.replace("Ġ", "")
            idx = self.vocab.get(word, None)
            s = 0.5 if idx is None else float(self.idf[idx])
            scores.append(s)
        v = torch.tensor(scores, dtype=torch.float32)
        mv = float(v.max().item()) if v.numel() else 1.0
        return v / mv if mv > 0 else v

    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = str(row[self.text_col])
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)
        gate = self._tfidf_gate_for_ids(input_ids)
        return {
            "id": str(row[self.id_col]),
            "input_ids": input_ids,
            "attention_mask": attn,
            "tfidf_gates": gate,
            "text": text,
        }

def collate_batch(batch):
    ids = [b["id"] for b in batch]
    texts = [b["text"] for b in batch]
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    tfidf_gates = torch.stack([b["tfidf_gates"] for b in batch])
    return {
        "ids": ids,
        "texts": texts,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tfidf_gates": tfidf_gates,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to cleaned Twitter Parquet (e.g., out_im1/data.parquet)")
    ap.add_argument("--output", required=True, help="Path to write predictions CSV")
    ap.add_argument("--rows", type=int, default=20000, help="How many rows to run (subset for CPU)")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--ckpt", default="checkpoint_esa_tfidf_debug.pt", help="Model checkpoint path")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--model_name", default="distilroberta-base")
    args = ap.parse_args()

    print("Loading data:", args.input)
    df = pd.read_parquet(args.input)
    df = df[[args.id_col, args.text_col]].dropna().drop_duplicates(subset=[args.id_col])
    if args.rows:
        df = df.head(args.rows)
    print("Rows to infer:", len(df))

    dataset = TwitterSet(df, text_col=args.text_col, id_col=args.id_col, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_batch)

    print("Loading model:", args.model_name)
    model = EmotionAwareRoBERTa(model_name=args.model_name, num_labels=len(LABELS), dropout=0.0).to(DEVICE)
    sd = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)
    model.eval()

    all_rows = []
    with torch.no_grad():
        for step, batch in enumerate(loader, 1):
            ids = batch["ids"]
            texts = batch["texts"]
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            tfidf_gates = batch["tfidf_gates"].to(DEVICE)

            out = model(input_ids, attention_mask, tfidf_gates)
            probs = softmax(out["logits"], dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            scores = probs.max(axis=1)

            for i in range(len(ids)):
                all_rows.append({
                    "id": ids[i],
                    "text": texts[i],
                    "pred_label_id": int(preds[i]),
                    "pred_label": LABELS[int(preds[i])] if int(preds[i]) < len(LABELS) else f"label_{int(preds[i])}",
                    "pred_score": float(scores[i]),
                })

            if step % 100 == 0:
                print(f"  processed {step*args.batch}/{len(dataset)}")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print("Saved predictions ->", args.output)

if __name__ == "__main__":
    main()
