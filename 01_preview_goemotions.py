# 01_preview_goemotions.py
from datasets import load_dataset

# Load the simplified single-label GoEmotions split (good starting point)
ds = load_dataset("go_emotions", "simplified")

# Basic info
print("Splits:", {k: len(v) for k, v in ds.items()})
print("Columns:", ds["train"].column_names)
print("Example row:", ds["train"][0])

# Try to get human-readable label names (handles different dataset schemas)
label_names = None
try:
    # Most configs: labels is a Sequence(ClassLabel)
    label_names = ds["train"].features["labels"].feature.names
except Exception:
    try:
        # Some configs: labels is a ClassLabel
        label_names = ds["train"].features["labels"].names
    except Exception:
        label_names = None

print("Label set size:", None if label_names is None else len(label_names))
if label_names is not None:
    print("First 10 labels:", label_names[:10])
else:
    print("Label names not provided by this config. We'll map IDs directly.")
