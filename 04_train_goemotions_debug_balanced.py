# 04_train_goemotions_debug_balanced.py
import time, torch, numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from transformers import get_linear_schedule_with_warmup

from collator_tfidf import CollatorTFIDF, tfidf, tok
from model_esa import EmotionAwareRoBERTa

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 2               # small for CPU
LR = 2e-5
MAX_LEN = 128
NUM_LABELS = 28
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)

# --- load tiny-but-bigger subset for speed ---
ds_full = load_dataset("go_emotions", "simplified")
train_ds_small = ds_full["train"].select(range(3000))
val_ds   = ds_full["validation"].select(range(512))
test_ds  = ds_full["test"].select(range(512))

# --- oversample train to reduce class collapse ---
y_small = np.array([
    ex["labels"][0] if isinstance(ex["labels"], list) else ex["labels"]
    for ex in train_ds_small
])
idx = np.arange(len(train_ds_small)).reshape(-1, 1)
ros = RandomOverSampler(random_state=SEED)
idx_res, y_res = ros.fit_resample(idx, y_small)
train_ds = train_ds_small.select(idx_res.flatten().tolist())

# --- collator & loaders ---
collator = CollatorTFIDF(tok, tfidf, max_len=MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collator)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

# --- model ---
model = EmotionAwareRoBERTa(model_name="distilroberta-base", num_labels=NUM_LABELS, dropout=0.3).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * max(1, len(train_loader))
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

# --- class-weighted loss (use oversampled labels) ---
classes = np.arange(NUM_LABELS)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_res)
class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

def run_epoch(loader, train: bool):
    model.train(train)
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for i, batch in enumerate(loader):
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            tfidf_gates=batch["tfidf_gates"],
            labels=None  # we compute weighted loss manually
        )["logits"]
        loss = loss_fn(logits, batch["labels"])

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += float(loss.item()); n += 1
        preds = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
        trues = batch["labels"].detach().cpu().numpy().tolist()
        all_preds.extend(preds); all_labels.extend(trues)

        if (i+1) % 100 == 0:
            print(f"  step {i+1}/{len(loader)} | loss={total_loss/max(n,1):.4f}")

    return total_loss / max(n,1), f1_score(all_labels, all_preds, average="macro")

def evaluate(loader, title="Validation"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                tfidf_gates=batch["tfidf_gates"],
                labels=None
            )["logits"]
            preds = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            trues = batch["labels"].detach().cpu().numpy().tolist()
            all_preds.extend(preds); all_labels.extend(trues)
    print(f"\n{title} report:\n", classification_report(all_labels, all_preds, digits=4))
    return f1_score(all_labels, all_preds, average="macro")

print(f"Device: {DEVICE} | Tiny(train) steps ≈ {len(train_loader)}")
t0 = time.time()
tr_loss, tr_f1 = run_epoch(train_loader, train=True)
val_f1 = evaluate(val_loader, title="Validation")
print(f"\nBalanced tiny run | train_loss={tr_loss:.4f} train_macroF1={tr_f1:.4f} "
      f"val_macroF1={val_f1:.4f} | {(time.time()-t0)/60:.1f} min")

print("\nTiny Test evaluation:")
_ = evaluate(test_loader, title="Test")

# --- save checkpoint for inference on Twitter data ---
torch.save(model.state_dict(), "checkpoint_esa_tfidf_debug.pt")
print("Saved -> checkpoint_esa_tfidf_debug.pt")
