import torch
import pickle
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import torch.nn as nn
import multiprocessing
from datetime import datetime
import matplotlib.pyplot as plt
from scripts.model import CHRTModel
from torch.utils.data import DataLoader
from scripts.config import cfg, update_version
from scripts.logger import get_logger, global_exception_logger
from scripts.mlm_data import MLMDataset, DynamicPaddingCollator

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

def plot_metrics(log, out_dir):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = log["epoch"]

    ax1.plot(x, log["train_loss"], label="Train Loss", linewidth=2, marker='o')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, log["train_acc"], label="Train Accuracy", linewidth=2,
             marker='o', color='green')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training Accuracy", fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png", dpi=150)
    plt.close()

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return True
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def compute_metrics(logits, labels):
    preds = logits.argmax(dim=-1)
    mask = (labels != -100)

    if mask.sum() == 0:
        return 0.0, 0

    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    accuracy = correct / total
    return accuracy, total

def train_epoch(model, dataloader, optimizer, criterion, scaler, device, grad_accum_steps, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc="Training", ncols=120, leave=True)

    for step, batch in enumerate(pbar, 1):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device, enabled=torch.cuda.is_available() and "cuda" in str(device)):
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if step % cfg.print_every == 0:
            with torch.no_grad():
                acc, n_tokens = compute_metrics(logits.detach(), labels)
                total_correct += acc * n_tokens
                total_tokens += n_tokens
        else:
            total_tokens += (labels != -100).sum().item()

        total_loss += loss.item() * grad_accum_steps

        if step % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % cfg.print_every == 0:
            current_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
            pbar.set_postfix({
                "loss": f"{loss.item() * grad_accum_steps:.4f}",
                "acc": f"{current_acc:.4f}",
                "tokens": f"{total_tokens:,}"
            })

    if len(dataloader) % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_acc, total_tokens

def main():
    if multiprocessing.current_process().name == "MainProcess":
        update_version("mlm")

        NAME = cfg.templates.name.format(
            ver=cfg.ver.mlm,
            mode="mlm",
            date=datetime.now().strftime(cfg.templates.date)
        )

        DIR = cfg.paths.outputs / NAME
        DIR.mkdir(parents=True, exist_ok=True)
        logger = get_logger(__name__, log_file=DIR / "training.log")
        global_exception_logger(logger)
    else:
        logger = get_logger(__name__)
        NAME = cfg.templates.name.format(
            ver=cfg.ver.mlm,
            mode="mlm",
            date=datetime.now().strftime(cfg.templates.date)
        )
        DIR = cfg.paths.outputs / NAME
    globals().update({"NAME": NAME, "DIR": DIR, "logger": logger})

    logger.info("=" * 70)
    logger.info(f"Starting MLM: {NAME}")
    logger.info("=" * 70)

    logger.info(f"Device: {cfg.device}")
    logger.info(f"Random seed: {cfg.seed}")
    logger.info(f"Mixed precision training: {torch.cuda.is_available() and 'cuda' in str(cfg.device)}")

    logger.info("--- Hyperparameters ---")
    logger.info(f"Batch size: {cfg.mlm.batch_size}")
    logger.info(f"Gradient accumulation steps: {cfg.grad_accum_steps}")
    logger.info(f"Effective batch size: {cfg.mlm.batch_size * cfg.grad_accum_steps}")
    logger.info(f"Learning rate: {cfg.mlm.lr}")
    logger.info(f"Epochs: {cfg.mlm.epochs}")
    logger.info(f"Masking probability: {cfg.mlm.mask_prob}")
    logger.info(f"Patience: {cfg.patience}")
    logger.info(f"Warmup fraction: {cfg.warmup_frac}")

    logger.info("--- Loading Data ---")
    logger.info(f"Loading tokenizer from {cfg.paths.tokenizer}")
    with open(cfg.paths.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size:,}")

    logger.info(f"Loading corpus from {cfg.paths.corpus}")
    with open(cfg.paths.corpus, encoding='utf-8') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    logger.info(f"Corpus size: {len(smiles_list):,} reactions")

    logger.info("--- Model Architecture ---")
    logger.info(f"Embedding dimension: {cfg.emb_dim}")
    logger.info(f"Number of attention heads: {cfg.n_heads}")
    logger.info(f"Number of transformer layers: {cfg.n_layers}")
    logger.info(f"Feed-forward dimension: {cfg.ff_dim}")
    logger.info(f"Dropout: {cfg.dropout}")
    logger.info(f"Max sequence length: {cfg.max_len}")

    model = CHRTModel(
        vocab_size=tokenizer.vocab_size,
        emb_dim=cfg.emb_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        use_swiglu=cfg.use_swiglu,
        use_rms_norm=cfg.use_rms_norm,
        tie_word_embeddings=False
    ).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("--- Preparing DataLoader ---")
    dataset = MLMDataset(smiles_list, tokenizer)

    collator = DynamicPaddingCollator(
        pad_id=tokenizer.token_to_id[tokenizer.pad_token]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.mlm.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
        persistent_workers=True,
        prefetch_factor=cfg.prefetch_factor,
        drop_last=True
    )
    logger.info(f"Number of batches per epoch: {len(dataloader):,}")

    logger.info("--- Setting up Optimizer ---")
    no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': cfg.cls.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    use_fused = torch.cuda.is_available() and hasattr(torch.optim.AdamW, 'fused')
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=cfg.mlm.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=use_fused if use_fused else False
    )
    if use_fused:
        logger.info("Using fused AdamW optimizer")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    total_steps = len(dataloader) * cfg.mlm.epochs // cfg.grad_accum_steps
    warmup_steps = int(total_steps * cfg.warmup_frac)

    logger.info(f"Total training steps: {total_steps:,}")
    logger.info(f"Warmup steps: {warmup_steps:,}")

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    amp_enabled = torch.cuda.is_available() and "cuda" in str(cfg.device)
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    early_stopping = EarlyStopping(patience=cfg.patience, mode='max')

    logger.info("=" * 70)
    logger.info("Starting Training")
    logger.info("=" * 70)

    logs = []
    best_acc = 0.0

    for epoch in range(1, cfg.mlm.epochs + 1):
        logger.info(f"Epoch {epoch}/{cfg.mlm.epochs}")

        train_loss, train_acc, total_tokens = train_epoch(
            model, dataloader, optimizer, criterion, scaler,
            cfg.device, cfg.grad_accum_steps
        )

        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Train Accuracy: {train_acc:.4f}")
        logger.info(f"  Processed Tokens: {total_tokens:,}")
        logger.info(f"  Learning Rate: {current_lr:.2e}")

        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "lr": current_lr
        })

        if train_acc > best_acc:
            best_acc = train_acc
            ckpt_path = DIR / f"{NAME}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'config': dict(cfg)
            }, ckpt_path)
            logger.info(f"Saved best model (acc={best_acc:.4f})")

        improved = early_stopping(train_acc)
        if not improved:
            logger.info(f"No improvements. Patience: {early_stopping.counter}/{cfg.patience}")

        if early_stopping.early_stop:
            logger.info("Early stop")
            break

        scheduler.step()

    final_ckpt_path = DIR / f"{NAME}_final.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
    }, final_ckpt_path)
    logger.info(f"Saved final model -> {final_ckpt_path}")

    # Save training logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(DIR / "training_logs.csv", index=False)
    logger.info(f"Training logs -> {DIR / 'training_logs.csv'}")

    plot_metrics(log_df, DIR)
    logger.info(f"Training plots -> {DIR / 'metrics.png'}")

    logger.info("=" * 70)
    logger.info("Training Completed")
    logger.info(f"Best accuracy: {best_acc:.4f}")
    logger.info(f"Total epochs: {epoch}")
    logger.info(f"Output directory: {DIR}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()