import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import warnings
from pydantic import warnings as pydantic_warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
warnings.filterwarnings("ignore", category=pydantic_warnings.UnsupportedFieldAttributeWarning)
import math
import wandb
import torch
torch.cuda.empty_cache()
import pickle
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup

from scripts.model import CHRTModel
from scripts.modelema import ModelEMA
from scripts.config import cfg, update_version
from scripts.earlystopping import EarlyStopping
from scripts.cls_data import ClsDataset, SimpleCollator
from scripts.logger import get_logger, global_exception_logger
from scripts.functions import setup_env, setup_ddp, cleanup_ddp, choose_mlm_checkpoint

setup_env()

def train_epoch(logger, model, ema, dataloader, optimizer, scheduler,
                criterion, scaler, device, grad_accum_steps, epoch,
                rank=0, max_grad_norm=1.0, use_wandb=False):
    model.train()
    total_loss = 0.0
    all_preds, all_trues = [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (GPU {rank})", ncols=120, disable=(rank != 0))
    batch: dict
    for step, batch in enumerate(pbar, 1):
        ids = batch['input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
            pooled = model.module.sequence_repr(ids, mask)
            logits = model.module.classifier(pooled)
            loss = criterion(logits, labels)

        loss = loss / grad_accum_steps

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss at step {step}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        with torch.no_grad():
            preds = logits.argmax(dim=1).detach().cpu().tolist()
            all_preds.extend(preds)
            all_trues.extend(labels.detach().cpu().tolist())

        if step % grad_accum_steps == 0 or step == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if ema is not None:
                ema.update(model.module)

        total_loss += loss.item() * grad_accum_steps
        current_lr = scheduler.get_last_lr()[0]

        if step % 50 == 0:
            pbar.set_postfix({
                'loss': f"{loss.item() * grad_accum_steps:.4f}",
                'acc': f"{accuracy_score(all_trues[-1000:], all_preds[-1000:]):.4f}" if len(all_preds) > 100 else "N/A",
                'lr': f"{current_lr:.2e}"
            })

            if use_wandb and rank == 0:
                wandb.log({
                    "train/loss_step": loss.item() * grad_accum_steps,
                    "train/lr_step": current_lr,
                    "step": (epoch - 1) * len(dataloader) + step
                })

    avg_loss = total_loss / len(dataloader)
    train_acc = accuracy_score(all_trues, all_preds)
    return avg_loss, train_acc

def validate_epoch(model, dataloader, criterion, device, rank=0):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []

    pbar = tqdm(dataloader, desc=f"Validation (GPU {rank})", ncols=100, disable=(rank != 0))

    with torch.no_grad():
        for batch in pbar:
            ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            pooled = model.sequence_repr(ids, mask)
            logits = model.classifier(pooled)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_trues.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    val_acc = accuracy_score(all_trues, all_preds)
    val_f1 = f1_score(all_trues, all_preds, average='macro', zero_division=0)
    return avg_loss, val_acc, val_f1

def main_worker(rank, world_size):
    setup_ddp(rank, world_size, port='12357')
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    use_wandb = (rank == 0 and cfg.wandb.enabled)

    NAME = None
    DIR = None

    if rank == 0:
        update_version('cls')
        now = datetime.now().strftime(cfg.templates.date)
        NAME = cfg.templates.name.format(ver=cfg.ver.cls, mode='cls', date=now)
        DIR = cfg.paths.outputs / NAME
        DIR.mkdir(parents=True, exist_ok=True)

        logger = get_logger(__name__, log_file=DIR / "training.log")
        global_exception_logger(logger)

        logger.info("=" * 70)
        logger.info(f"Starting Supervised CLS Training: {NAME}")
        logger.info(f"Rank: {rank}/{world_size}")
        logger.info("=" * 70)
        logger.info(f"Device: {device}")
        logger.info(f"Random seed: {cfg.seed}")
        logger.info(f"Mixed precision: Enabled")
        logger.info("--- Training Hyperparameters ---")
        logger.info(f"Batch size per GPU: {cfg.cls.batch_size}")
        logger.info(f"Gradient accumulation: {cfg.grad_accum_steps}")
        logger.info(f"Effective batch size: {cfg.cls.batch_size * world_size * cfg.grad_accum_steps}")
        logger.info(f"Learning rate (encoder): {cfg.cls.lr_encoder}")
        logger.info(f"Learning rate (head): {cfg.cls.lr_head}")
        logger.info(f"Epochs: {cfg.cls.epochs}")
        logger.info(f"Number of classes: {cfg.num_classes}")
    else:
        logger = get_logger(__name__)

    logger.info(f"Loading tokenizer from {cfg.paths.tokenizer}")
    with open(cfg.paths.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size:,}")

    logger.info(f"Loading labeled data from {cfg.paths.cls_train}")
    df_train = pd.read_csv(cfg.paths.cls_train)
    logger.info(f"Loading validation data from {cfg.paths.cls_val}")
    df_val = pd.read_csv(cfg.paths.cls_val)

    logger.info(f"Train: {len(df_train):,}")
    logger.info(f"Val:   {len(df_val):,}")

    train_dataset = ClsDataset(df_train, tokenizer)
    val_dataset = ClsDataset(df_val, tokenizer)

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    collator = SimpleCollator(pad_id=tokenizer.token_to_id[tokenizer.pad_token])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.cls.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.cls.batch_size,
        sampler=val_sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor
    )

    if rank == 0:
        logger.info(f"Batches per epoch: train={len(train_loader):,}, val={len(val_loader):,}")
        logger.info("--- Model Architecture ---")
        logger.info(f"Embedding dim: {cfg.emb_dim}")
        logger.info(f"Attention heads: {cfg.n_heads}")
        logger.info(f"Transformer layers: {cfg.n_layers}")
        logger.info(f"Feed-forward dim: {cfg.ff_dim}")
        logger.info(f"Dropout: {cfg.dropout}")
        logger.info(f"Max sequence length: {cfg.max_len}")

    mlm_ckpt = choose_mlm_checkpoint(cfg.paths.outputs)
    logger.info(f"Loading MLM weights from: {mlm_ckpt}")

    model = CHRTModel(
        vocab_size=tokenizer.vocab_size,
        emb_dim=cfg.emb_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        drop_path_rate=cfg.drop_path_rate,
        cls_dropout=cfg.cls.dropout,
        num_classes=cfg.num_classes
    ).to(device)

    mlm_state = torch.load(mlm_ckpt, map_location=device, weights_only=True)
    if 'model_state' in mlm_state:
        mlm_state = mlm_state['model_state']

    missing, unexpected = model.load_state_dict(mlm_state, strict=False)
    if rank == 0 and (missing or unexpected):
        logger.info(f"Missing keys: {len(missing)}")
        logger.info(f"Unexpected keys: {len(unexpected)}")
    logger.info("MLM weights loaded successfully")

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable:,}")

    ema = None
    if rank == 0:
        ema = ModelEMA(model.module, decay=cfg.ema_decay, device=device)

    if rank == 0 and use_wandb:
        wandb.init(
            project="chrt-cls",
            name=NAME,
            config={
                "vocab_size": tokenizer.vocab_size,
                "emb_dim": cfg.emb_dim,
                "n_heads": cfg.n_heads,
                "n_layers": cfg.n_layers,
                "ff_dim": cfg.ff_dim,
                "dropout": cfg.dropout,
                "max_len": cfg.max_len,
                "num_classes": cfg.num_classes,
                "batch_size": cfg.cls.batch_size,
                "lr_encoder": cfg.cls.lr_encoder,
                "lr_head": cfg.cls.lr_head,
                "epochs": cfg.cls.epochs,
                "train_size": len(df_train),
                "val_size": len(df_val),
            }
        )
        logger.info(f"WandB initialized: {wandb.run.url}")

    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name or 'attn_pool' in name or 'cls_dropout' in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': cfg.cls.lr_encoder},
        {'params': head_params, 'lr': cfg.cls.lr_head}
    ],
        weight_decay=cfg.cls.weight_decay,
        fused=True
    )

    if cfg.cls.use_class_weights:
        class_counts = df_train['label'].value_counts().sort_index().values
        total = class_counts.sum()
        class_weights = total / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
        if rank == 0:
            logger.info("Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, cfg.grad_accum_steps))
    num_training_steps = steps_per_epoch * cfg.cls.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_frac * num_training_steps),
        num_training_steps=num_training_steps
    )

    if rank == 0:
        logger.info(f"Total training steps: {num_training_steps:,}")
        logger.info(f"Warmup steps: {int(cfg.warmup_frac * num_training_steps):,}")

    scaler = torch.amp.GradScaler()
    early_stopping = EarlyStopping(patience=cfg.patience)

    if rank == 0:
        logger.info("=" * 70)
        logger.info("Starting Training")
        logger.info("=" * 70)

    logs = []
    best_val_acc = -1.0

    epoch = 0
    eval_model = model.module

    train_loss = train_acc = None
    val_loss = val_acc = val_f1 = None

    for epoch in range(1, cfg.cls.epochs + 1):
        sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss, train_acc = train_epoch(
            logger, model, ema, train_loader, optimizer, scheduler,
            criterion, scaler, device, cfg.grad_accum_steps, epoch,
            rank=rank, max_grad_norm=cfg.max_grad_norm, use_wandb=use_wandb
        )

        if rank == 0 and ema is not None:
            eval_model = ema.get_model()
        else:
            eval_model = model.module

        val_loss, val_acc, val_f1 = validate_epoch(
            eval_model, val_loader, criterion, device, rank=rank
        )

        current_lr = scheduler.get_last_lr()[0]

        if rank == 0:
            logger.info(f"Epoch {epoch}/{cfg.cls.epochs} Summary:")
            logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            logger.info(f"  LR: {current_lr:.2e}")

            logs.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "lr": current_lr
            })

            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss_epoch": train_loss,
                    "train/accuracy_epoch": train_acc,
                    "val/loss_epoch": val_loss,
                    "val/accuracy_epoch": val_acc,
                    "val/f1_epoch": val_f1,
                    "train/lr_epoch": current_lr,
                    "early_stopping/counter": early_stopping.counter,
                    "early_stopping/best_acc": early_stopping.best_score if early_stopping.best_score else val_acc
                })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = DIR / f"{NAME}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state': eval_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'ema_updates': ema.updates if ema else 0,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'config': {
                        "vocab_size": tokenizer.vocab_size,
                        "emb_dim": cfg.emb_dim,
                        "n_heads": cfg.n_heads,
                        "n_layers": cfg.n_layers,
                        "ff_dim": cfg.ff_dim,
                        "dropout": cfg.dropout,
                        "max_len": cfg.max_len,
                        "num_classes": cfg.num_classes,
                    }
                }, ckpt_path)
                logger.info(f"Saved best model (val_acc={best_val_acc:.4f})")

                if use_wandb:
                    wandb.run.summary["best_val_accuracy"] = best_val_acc
                    wandb.run.summary["best_val_f1"] = val_f1
                    wandb.run.summary["best_epoch"] = epoch

            improved = early_stopping(val_acc)
            if not improved:
                logger.info(f"No improvement: {early_stopping.counter}/{cfg.patience}")

            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                if use_wandb:
                    wandb.log({"early_stopped": True, "stopped_at_epoch": epoch})
                break

    if rank == 0:
        final_ckpt_path = DIR / f"{NAME}_final.pt"
        torch.save({
            'epoch': epoch,
            'model_state': eval_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
        }, final_ckpt_path)
        logger.info(f"Saved final model -> {final_ckpt_path}")

        log_df = pd.DataFrame(logs)
        log_df.to_csv(DIR / "training_logs.csv", index=False)

        if use_wandb:
            wandb.log({"training_logs": wandb.Table(dataframe=log_df)})

            loss_df = log_df[["epoch", "train_loss", "val_loss"]]
            wandb.log({
                "charts/loss_curves": wandb.plot.line(
                    wandb.Table(dataframe=loss_df),
                    "epoch", "train_loss", title="Loss Curves"
                )
            })

            acc_df = log_df[["epoch", "train_acc", "val_acc"]]
            wandb.log({
                "charts/accuracy_curves": wandb.plot.line(
                    wandb.Table(dataframe=acc_df),
                    "epoch", "train_acc", title="Accuracy Curves"
                )
            })
            wandb.finish()

        logger.info("=" * 70)
        logger.info("Supervised Training")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        logger.info(f"Total epochs: {epoch}")
        logger.info(f"Output directory: {DIR}")
        logger.info("=" * 70)
    cleanup_ddp()

if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPUs available")
    print(f"Starting DDP Supervised CLS training with {num_gpus} GPUs")
    mp.spawn(main_worker, args=(num_gpus,), nprocs=num_gpus)