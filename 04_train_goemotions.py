# 04_train_goemotions.py
import math, time, torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler
from transformers import get_linear_schedule_with_warmup

from collator_tfidf import CollatorTFIDF, tfidf, tok
from model_esa import EmotionAwareRoBERTa

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-5
MAX_LEN = 128
NUM_LABELS = 28
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Load dataset ---
ds = load_dataset("go_emotions", "simplified")

# Oversample train to mitigate class imbalance
train_texts = [ex["text"] for ex in ds["train"]]
train_labels = [ex["labels"][0] for ex in ds["train"]]  # single label per row in this config

X = np.arange(len(train_texts)).reshape(-1, 1)
y = np.array(train_labels)
ros = RandomOverSampler(random_state=SEED)
X_res, y_res = ros.fit_resample(X, y)
res_idx = X_res.flatten().tolist()

train_ds = ds["train"].select(res_idx)
val_ds = ds["validation"]
test_ds = ds["test"]

# --- Collator & loaders ---
collator = CollatorTFIDF(tok, tfidf, max_len=MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collator)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

# --- Model, optimizer, scheduler ---
model = EmotionAwareRoBERTa(num_labels=NUM_LABELS, dropout=0.3).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * max(1, len(train_loader))
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

def run_epoch(loader, train: bool):
    model.train(train)
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            tfidf_gates=batch["tfidf_gates"],
            labels=batch["labels"].squeeze(-1)  # [[id]] -> [id]
        )
        loss = out["loss"]
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += float(loss.item())
        n += 1

        preds = out["logits"].argmax(dim=-1).detach().cpu().numpy().tolist()
        trues = batch["labels"].squeeze(-1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(trues)

    avg_loss = total_loss / max(n, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1

def evaluate(loader, title="Validation"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                tfidf_gates=batch["tfidf_gates"]
            )
            preds = out["logits"].argmax(dim=-1).detach().cpu().numpy().tolist()
            trues = batch["labels"].squeeze(-1).detach().cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(trues)
    print(f"\n{title} report:\n", classification_report(all_labels, all_preds, digits=4))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return macro_f1

print(f"Device: {DEVICE} | Train steps/epoch ≈ {len(train_loader)}")

best_val = -1.0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_f1 = run_epoch(train_loader, train=True)
    val_f1 = evaluate(val_loader, title="Validation")
    dt = time.time() - t0
    print(f"\nEpoch {epoch}/{EPOCHS} | train_loss={tr_loss:.4f} train_macroF1={tr_f1:.4f} "
          f"val_macroF1={val_f1:.4f} | {dt/60:.1f} min")

    if val_f1 > best_val:
        best_val = val_f1
        torch.save(model.state_dict(), "checkpoint_esa_tfidf.pt")
        print("Saved best checkpoint -> checkpoint_esa_tfidf.pt")

print("\nFinal Test evaluation:")
_ = evaluate(test_loader, title="Test")
