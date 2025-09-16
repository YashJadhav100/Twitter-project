# 03_model_esa.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ESA(nn.Module):
    """
    Emotion-Specific Attention-ish module:
    - Learns a token-wise gate (salience) from hidden states
    - Learns a feature-wise scale vector
    """
    def __init__(self, hidden_size):
        super().__init__()
        mid = max(32, hidden_size // 2)
        self.token_gate = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.Tanh(),
            nn.Linear(mid, hidden_size),
            nn.Sigmoid()  # per-token, per-dim gate in [0,1]
        )
        self.scale = nn.Parameter(torch.ones(hidden_size))  # feature-wise scaling

    def forward(self, x):
        # x: (B, L, H)
        g = self.token_gate(x)       # (B, L, H)
        x = x * g                    # emphasize salient token features
        x = x * self.scale           # feature-dimension scaling
        return x

class EmotionAwareRoBERTa(nn.Module):
    """
    RoBERTa backbone + optional TF-IDF gating + ESA + pooled classifier.
    Designed for single-label classification (CrossEntropy) with K classes.
    """
    def __init__(self, model_name="roberta-base", num_labels=28, dropout=0.3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        H = self.config.hidden_size
        self.esa = ESA(H)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(H, num_labels)

    def forward(self, input_ids, attention_mask, tfidf_gates=None, labels=None):
        # Backbone
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state  # (B, L, H)

        # Optional TF-IDF token gate (shape: B,L)
        if tfidf_gates is not None:
            x = x * tfidf_gates.unsqueeze(-1)

        # ESA module
        x = self.esa(x)

        # Masked mean pool over tokens
        mask = attention_mask.unsqueeze(-1).float()  # (B,L,1)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / denom  # (B,H)

        logits = self.classifier(self.dropout(pooled))  # (B,K)

        loss = None
        if labels is not None:
            # labels expected as int class IDs (B,) — convert if [[id]]
            if labels.dim() == 2 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"logits": logits, "loss": loss}

if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("roberta-base")
    model = EmotionAwareRoBERTa(num_labels=28)

    texts = ["I am thrilled about the news!", "This is so frustrating."]
    enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    gates = torch.ones_like(enc["input_ids"], dtype=torch.float32)  # dummy all-ones gate
    out = model(enc["input_ids"], enc["attention_mask"], tfidf_gates=gates)
    print("logits shape:", out["logits"].shape)
