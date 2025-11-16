import math
import time
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scripts.model import CHRTModel
from config import cfg, update_version
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_mem_efficient_sdp(True)

update_version("cls")

NAME = cfg.templates.name.format(
    ver=cfg.ver.cls,
    mode="cls",
    date=datetime.now().strftime(cfg.templates.date)
)

DIR = cfg.paths.outputs / NAME
DIR.mkdir(parents=True, exist_ok=True)

class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.ema.to(device=device)
        self.updates = 0
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        self.updates += 1
        with torch.no_grad():
            d = self.decay * (1 - math.exp(-self.updates / 2000))
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def get_model(self):
        return self.ema

class ClsDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.smiles = df['smiles'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.token_to_id[tokenizer.pad_token]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        label = self.labels[idx]
        seq = self.tokenizer.encode(smiles, max_length=None)
        return {
            'input_ids': seq,
            'labels': int(label)
        }

def collate_batch(samples, pad_id):
    max_len = max(len(s['input_ids']) for s in samples)
    ids, mask, labels = [], [], []
    for s in samples:
        seq = s['input_ids']
        pad_n = max_len - len(seq)
        ids.append(seq + [pad_id]*pad_n)
        mask.append([1]*len(seq) + [0]*pad_n)
        labels.append(s['labels'])
    return {
        'input_ids': torch.tensor(ids, dtype=torch.long),
        'attention_mask': torch.tensor(mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }

@dataclass
class DynamicPadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
    def __call__(self, samples):
        return collate_batch(samples, self.pad_id)

def plot_metrics(log_df):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    x = log_df["epoch"]

    plt.plot(x, log_df["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(x, log_df["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    plt.plot(x, log_df["train_acc"], label="Train Acc", linewidth=2)
    plt.plot(x, log_df["val_acc"], label="Val Acc", linewidth=2, linestyle="--")
    plt.plot(x, log_df["val_f1"], label="Val Macro-F1", linewidth=2, linestyle=":")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"CHRT CLS Model Training Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIR / "metrics.png")
    plt.close()

def choose_mlm_checkpoint(base_dir) -> Path:
    mlm_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and "mlm" in p.name])
    if not mlm_dirs:
        raise FileNotFoundError("MLM models not found")

    print("\nAvailable MLM models:")
    for i, d in enumerate(mlm_dirs, 1):
        pt_files = list(d.glob("*.pt"))
        ckpt = pt_files[0].name if pt_files else "no .pt found"
        print(f"[{i}] {ckpt}")

    choice = int(input("\nSelect MLM model [number]: ")) - 1
    chosen_dir = mlm_dirs[choice]
    ckpt_files = list(chosen_dir.glob("*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt file in {chosen_dir}")
    return ckpt_files[0]

def main():
    with open(cfg.paths.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)

    df_train = pd.read_csv(cfg.paths.cls_train_full)
    df_val = pd.read_csv(cfg.paths.cls_val)

    train_dataset = ClsDataset(df_train, tokenizer)
    val_dataset = ClsDataset(df_val, tokenizer)
    train_labels = df_train['label'].tolist()

    counts = np.bincount(train_labels, minlength=cfg.num_classes)
    class_counts = torch.tensor(counts, dtype=torch.float)
    class_weights = 1.0 / torch.clamp(class_counts, min=1.0)

    labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    sample_weights = class_weights[labels_tensor]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))

    train_collate = DynamicPadCollator(train_dataset.pad_id)
    val_collate = DynamicPadCollator(val_dataset.pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.cls.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        collate_fn=train_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.cls.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        collate_fn=val_collate
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    model = CHRTModel(
        vocab_size=tokenizer.vocab_size,
        emb_dim=cfg.emb_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        use_swiglu=cfg.use_swiglu,
        use_rms_norm=cfg.use_rms_norm
    )

    mlm_ckpt = choose_mlm_checkpoint(cfg.paths.outputs)
    print(f"\nLoading MLM weights from: {mlm_ckpt}")
    state = torch.load(mlm_ckpt, map_location=cfg.device)

    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.classifier = model.get_classifier_head(num_classes=cfg.num_classes)
    model.to(cfg.device)

    #try:
    #    print("Compiling model with torch.compile...")
    #    model = torch.compile(model, mode='max-autotune')
    #    print("Model compiled successfully")
    #except Exception as e:
    #    print(e)

    if cfg.ema.enabled:
        ema = ModelEMA(model, decay=cfg.ema.decay, device=cfg.device)
        print(f"EMA initialized with decay={cfg.ema.decay}")
    else:
        ema = None
        print("EMA disabled")

    enc_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    head_params = list(model.classifier.parameters())

    optim_groups = [
        {"params": enc_params, "lr": cfg.cls.lr_encoder},
        {"params": head_params, "lr": cfg.cls.lr_head},
    ]
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=cfg.cls.weight_decay, fused=True if torch.cuda.is_available() else False)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, cfg.grad_accum_steps))
    num_training_steps = steps_per_epoch * cfg.cls.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_frac * num_training_steps),
        num_training_steps=num_training_steps
    )

    try:
        from torch import amp
        autocast = lambda enabled: amp.autocast('cuda', enabled=enabled)
        GradScaler = lambda enabled: amp.GradScaler(enabled=enabled)
    except Exception:
        from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler(enabled=(cfg.device == 'cuda'))

    logs = []
    best_val_acc = -1.0
    patience_left = cfg.patience

    print(f"train size = {len(train_dataset)}, val size = {len(val_dataset)}")
    print(f"batches/epoch: train={len(train_loader)}, val={len(val_loader)}")
    print(f"optimizer steps per epoch (with accum={cfg.grad_accum_steps}): {steps_per_epoch}")

    for epoch in range(1, cfg.cls.epochs + 1):
        model.train()
        total_loss = 0.0
        all_preds, all_trues = [], []

        t0 = time.perf_counter()
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader, 1):
            ids = batch['input_ids'].to(cfg.device, non_blocking=True)
            mask = batch['attention_mask'].to(cfg.device, non_blocking=True)
            labels = batch['labels'].to(cfg.device, non_blocking=True)

            with autocast(enabled=(cfg.device == 'cuda')):
                pooled = model.sequence_repr(ids, mask)
                logits = model.classifier(pooled)
                loss = criterion(logits, labels) / max(1, cfg.grad_accum_steps)

            scaler.scale(loss).backward()

            if i % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            total_loss += loss.item() * max(1, cfg.grad_accum_steps)
            all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            all_trues.extend(labels.detach().cpu().tolist())

            if (i % cfg.print_every == 0) or (i == len(train_loader)):
                dt = time.perf_counter() - t0
                ips = i / max(dt, 1e-9)
                eta_min = (len(train_loader) - i) / max(ips, 1e-6) / 60.0
                print(f"[epoch {epoch}] {i}/{len(train_loader)} • {ips:.2f} it/s • ETA {eta_min:.1f} min", flush=True)

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_trues, all_preds)

        eval_model = ema.get_model() if ema is not None else model
        eval_model.eval()
        total_val_loss = 0.0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(cfg.device, non_blocking=True)
                mask = batch['attention_mask'].to(cfg.device, non_blocking=True)
                labels = batch['labels'].to(cfg.device, non_blocking=True)

                with autocast(enabled=(cfg.device == 'cuda')):
                    pooled = eval_model.sequence_repr(ids, mask)
                    logits = eval_model.classifier(pooled)
                    loss = criterion(logits, labels)

                total_val_loss += loss.item()
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_trues.extend(labels.cpu().tolist())

        val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_trues, all_preds)
        val_f1  = f1_score(all_trues, all_preds, average='macro')

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )

        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1":  val_f1
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_left = cfg.patience
            best_state = {
                "model_state": eval_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "ema_updates": ema.updates if ema is not None else 0,
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "emb_dim": cfg.emb_dim,
                    "n_heads": cfg.n_heads,
                    "n_layers": cfg.n_layers,
                    "ff_dim": cfg.ff_dim,
                    "dropout": cfg.dropout,
                    "max_len": cfg.max_len,
                    "num_classes": cfg.num_classes
                }
            }
            ckpt = DIR / f"{NAME}.pt"
            torch.save(best_state, ckpt)
            print(f"Saved best checkpoint to {ckpt} (val_acc={best_val_acc:.4f})")
        else:
            patience_left -= 1
            print(f"No improvement. Patience left: {patience_left}")
            if patience_left <= 0:
                print("Early stopping")
                break

    log = pd.DataFrame(logs)
    log.to_csv(DIR / "training_logs.csv", index=False)
    plot_metrics(log)

    print("Training complete. Best val_acc:", f"{best_val_acc:.4f}" if best_val_acc >= 0 else "n/a")

if __name__ == '__main__':
    main()