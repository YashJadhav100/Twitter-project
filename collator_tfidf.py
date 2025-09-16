# collator_tfidf.py
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# Load dataset
ds = load_dataset("go_emotions", "simplified")
train_texts = [x["text"] for x in ds["train"]]

# Fit TF-IDF on training set
tfidf = TfidfVectorizer(min_df=2, max_df=0.95)
tfidf.fit(train_texts)

# Tokenizer
tok = AutoTokenizer.from_pretrained("roberta-base")

class CollatorTFIDF:
    def __init__(self, tokenizer, tfidf, max_len=128):
        self.tokenizer = tokenizer
        self.tfidf = tfidf
        self.max_len = max_len
        self.vocab = tfidf.vocabulary_
        self.idf = tfidf.idf_

        # precompute special tokens
        self.specials = {
            tokenizer.cls_token,
            tokenizer.sep_token,
            tokenizer.pad_token,
        }

    def _as_rows(self, batch):
        # batch may be dict-of-lists (from dataset slicing) or list-of-dicts (DataLoader)
        if isinstance(batch, dict):
            keys = batch.keys()
            n = len(batch[next(iter(keys))])
            return [{k: batch[k][i] for k in keys} for i in range(n)]
        return batch

    def _to_class_id(self, lab):
        """
        Ensure a single integer class id per example.
        - If listlike (e.g., [27]) -> take first element
        - If longer list (rare) -> still take first as single-label target
        - If scalar -> cast to int
        """
        if isinstance(lab, (list, tuple)):
            return int(lab[0]) if len(lab) > 0 else 0
        return int(lab)

    def __call__(self, batch):
        rows = self._as_rows(batch)
        texts = [x["text"] for x in rows]
        labels = [self._to_class_id(x["labels"]) for x in rows]

        # Tokenize
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        # TF-IDF gating vector per sequence
        gates = []
        for ids in enc["input_ids"]:
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            scores = []
            for tok in tokens:
                if tok in self.specials:
                    scores.append(1.0)
                    continue
                word = tok.replace("Ġ", "")  # RoBERTa space marker
                idx = self.vocab.get(word, None)
                if idx is None:
                    s = 0.5  # unseen token: mid weight
                else:
                    s = float(self.idf[idx])
                scores.append(s)
            t = torch.tensor(scores, dtype=torch.float32)
            maxv = float(t.max().item()) if t.numel() > 0 else 1.0
            if maxv > 0:
                t = t / maxv
            gates.append(t)

        enc["tfidf_gates"] = torch.stack(gates)                    # (B, L)
        enc["labels"] = torch.tensor(labels, dtype=torch.long)     # (B,)
        return enc

# Quick test
if __name__ == "__main__":
    collator = CollatorTFIDF(tok, tfidf)
    small = ds["train"].select(range(2))
    sample = collator(small)
    print("input_ids shape:", sample["input_ids"].shape)
    print("tfidf_gates shape:", sample["tfidf_gates"].shape)
    print("labels:", sample["labels"])
