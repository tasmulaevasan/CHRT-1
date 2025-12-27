import warnings
from pydantic import warnings as pydantic_warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
warnings.filterwarnings("ignore", category=pydantic_warnings.UnsupportedFieldAttributeWarning)
import wandb
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from scripts.model import CHRTModel
from scripts.config import cfg, update_version
from scripts.earlystopping import EarlyStopping
from scripts.logger import get_logger, global_exception_logger
from scripts.mlm_data import MLMDataset, DynamicPaddingCollator
from scripts.functions import setup_env, setup_ddp, cleanup_ddp, choose_mlm_checkpoint

setup_env()

def select_training_mode(base_dir: Path):
    print("Select training mode:")
    print("[1] Start from scratch")
    print("[2] Resume from checkpoint")
    mode = 1

    if mode == 1:
        return None
    elif mode == 2:
        ckpt_path = choose_mlm_checkpoint(base_dir)
        print(f"Resuming from: {ckpt_path}")
        return ckpt_path
    else:
        raise ValueError("Invalid choice")

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    preds = logits.argmax(dim=-1)
    mask = (labels != -100)

    if mask.sum() == 0:
        return 0.0, 0

    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    accuracy = correct / total
    return accuracy, total

def train_epoch(logger, model, dataloader, optimizer, scheduler, criterion,
                scaler, device, grad_accum_steps, epoch, rank=0, max_grad_norm=1.0,
                use_wandb=False):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (GPU {rank})", ncols=120, disable=(rank != 0))

    for step, batch in enumerate(pbar, 1):
        batch: dict[str, torch.Tensor]
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=scaler.is_enabled()):
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss = loss / grad_accum_steps

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss at step {step}: {loss.item()}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        with torch.no_grad():
            acc_batch, n_tokens = compute_metrics(logits.detach(), labels)
            total_correct += acc_batch * n_tokens
            total_tokens += n_tokens
            total_loss += loss.item() * grad_accum_steps

        if step % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if not torch.isfinite(grad_norm):
                logger.error(f"Gradient norm is {grad_norm} at step {step}, skipping update")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if use_wandb and rank == 0:
                global_step = (epoch - 1) * len(dataloader) + step
                wandb.log({
                    "train/loss_step": loss.item() * grad_accum_steps,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "global_step": global_step
                })

        if step % cfg.print_every == 0:
            current_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
            pbar.set_postfix({
                "loss": f"{(loss.item() * grad_accum_steps):.4f}",
                "acc": f"{current_acc:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
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

def main_worker(rank, world_size):
    setup_ddp(rank, world_size, port='12355')
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    if rank == 0:
        checkpoint_to_load = select_training_mode(cfg.paths.outputs)
        ckpt_path_list = [checkpoint_to_load]
    else:
        ckpt_path_list = [None]

    dist.broadcast_object_list(ckpt_path_list, src=0)
    checkpoint_to_load = ckpt_path_list[0]

    if rank == 0:
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

        use_wandb = cfg.wandb.enabled
        if use_wandb:
            wandb.init(
                project="chrt-mlm",
                name=NAME,
                config={
                    "architecture": "CHRT-Transformer",
                    "dataset": "Chemical Reactions",
                    "emb_dim": cfg.emb_dim,
                    "n_heads": cfg.n_heads,
                    "n_layers": cfg.n_layers,
                    "ff_dim": cfg.ff_dim,
                    "dropout": cfg.dropout,
                    "batch_size": cfg.mlm.batch_size,
                    "grad_accum_steps": cfg.grad_accum_steps,
                    "effective_batch_size": cfg.mlm.batch_size * cfg.grad_accum_steps * world_size,
                    "lr": cfg.mlm.lr,
                    "epochs": cfg.mlm.epochs,
                    "mask_prob": cfg.mask_prob,
                    "world_size": world_size
                },
                dir=str(DIR),
                resume="allow" if checkpoint_to_load else None
            )
            logger.info(f"WandB initialized: {wandb.run.url}")
        else:
            logger.info("WandB disabled")
    else:
        NAME = cfg.templates.name.format(
            ver=cfg.ver.mlm,
            mode="mlm",
            date=datetime.now().strftime(cfg.templates.date)
        )
        DIR = cfg.paths.outputs / NAME
        logger = get_logger(__name__)
        use_wandb = False

    logger.info("=" * 70)
    logger.info(f"Starting MLM Training: {NAME} | Rank: {rank}/{world_size}")
    logger.info("=" * 70)

    if rank == 0:
        logger.info(f"Device: {cfg.device}")
        logger.info(f"Random seed: {cfg.seed}")
        logger.info(f"Mixed precision: Enabled")
        logger.info("--- Hyperparameters ---")
        logger.info(f"Batch size per GPU: {cfg.mlm.batch_size}")
        logger.info(f"Gradient accumulation: {cfg.grad_accum_steps}")
        logger.info(f"Effective batch size: {cfg.mlm.batch_size * cfg.grad_accum_steps * world_size}")
        logger.info(f"Learning rate: {cfg.mlm.lr}")
        logger.info(f"Epochs: {cfg.mlm.epochs}")
        logger.info(f"Masking probability: {cfg.mask_prob}")

    logger.info(f"Loading tokenizer from {cfg.paths.tokenizer}")
    with open(cfg.paths.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size:,}")

    logger.info(f"Loading corpus from {cfg.paths.corpus}")
    with open(cfg.paths.corpus, encoding='utf-8') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    logger.info(f"Corpus size: {len(smiles_list):,} reactions")

    if rank == 0:
        logger.info("--- Model Architecture ---")
        logger.info(f"Embedding dim: {cfg.emb_dim}")
        logger.info(f"Attention heads: {cfg.n_heads}")
        logger.info(f"Transformer layers: {cfg.n_layers}")
        logger.info(f"Feed-forward dim: {cfg.ff_dim}")
        logger.info(f"Dropout: {cfg.dropout}")
        logger.info(f"Max sequence length: {cfg.max_len}")

    model = CHRTModel(
        vocab_size=tokenizer.vocab_size,
        emb_dim=cfg.emb_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=cfg.max_len
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        if use_wandb:
            wandb.config.update({
                "total_params": total_params,
                "trainable_params": trainable_params
            })

    logger.info("--- Preparing DataLoader ---")
    dataset = MLMDataset(
        smiles_list,
        tokenizer,
        max_len=cfg.max_len,
        mask_prob=cfg.mask_prob
    )

    collator = DynamicPaddingCollator(
        pad_id=tokenizer.token_to_id[tokenizer.pad_token]
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        drop_last=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.mlm.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
        persistent_workers=True if cfg.num_workers > 0 else False
    )

    if rank == 0:
        logger.info(f"Batches per epoch: {len(dataloader):,}")

    logger.info("--- Setting up Optimizer ---")
    no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': cfg.mlm.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=cfg.mlm.lr,
        betas=(0.9, 0.999),
        fused=True
    )

    criterion = nn.CrossEntropyLoss()

    total_steps = len(dataloader) * cfg.mlm.epochs
    warmup_steps = int(total_steps * cfg.warmup_frac)

    if rank == 0:
        logger.info(f"Total training steps: {total_steps:,}")
        logger.info(f"Warmup steps: {warmup_steps:,}")

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 1

    if checkpoint_to_load:
        if rank == 0:
            logger.info(f"Loading checkpoint: {checkpoint_to_load}")

        map_location = {'cuda:0': f'cuda:{rank}'}
        checkpoint = torch.load(checkpoint_to_load, map_location=map_location)

        model.module.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1

        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}")

    scaler = torch.amp.GradScaler()
    early_stopping = EarlyStopping(patience=cfg.patience, mode='min')

    if rank == 0:
        logger.info("=" * 70)
        logger.info("Starting Training Loop")
        logger.info("=" * 70)

    logs = []
    best_acc = 0.0

    for epoch in range(start_epoch, cfg.mlm.epochs + 1):
        sampler.set_epoch(epoch)

        train_loss, train_acc, total_tokens = train_epoch(
            logger, model, dataloader, optimizer, scheduler, criterion,
            scaler, device, cfg.grad_accum_steps, epoch, rank=rank,
            use_wandb=use_wandb
        )

        current_lr = scheduler.get_last_lr()[0]

        if rank == 0:
            logger.info(f"Epoch {epoch}/{cfg.mlm.epochs} Summary:")
            logger.info(f"  Loss: {train_loss:.4f}")
            logger.info(f"  Accuracy: {train_acc:.4f}")
            logger.info(f"  Tokens: {total_tokens:,}")
            logger.info(f"  LR: {current_lr:.2e}")

            logs.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "lr": current_lr
            })

            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss_epoch": train_loss,
                    "train/accuracy_epoch": train_acc,
                    "train/tokens_processed": total_tokens,
                    "train/lr_epoch": current_lr,
                    "early_stopping/counter": early_stopping.counter,
                    "early_stopping/best_loss": early_stopping.best_score if early_stopping.best_score else train_loss
                })

            if train_acc > best_acc:
                best_acc = train_acc
                ckpt_path = DIR / f"{NAME}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'config': dict(cfg)
                }, ckpt_path)
                logger.info(f"Saved best model (acc={best_acc:.4f})")

                if use_wandb:
                    wandb.run.summary["best_accuracy"] = best_acc
                    wandb.run.summary["best_epoch"] = epoch

            improved = early_stopping(train_loss)
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
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
        }, final_ckpt_path)
        logger.info(f"Saved final model -> {final_ckpt_path}")

        log_df = pd.DataFrame(logs)
        log_df.to_csv(DIR / "training_logs.csv", index=False)

        if use_wandb:
            wandb.log({"training_logs": wandb.Table(dataframe=log_df)})

            wandb.log({
                "charts/loss_curve": wandb.plot.line(
                    wandb.Table(dataframe=log_df[["epoch", "train_loss"]]),
                    "epoch", "train_loss", title="Training Loss"
                ),
                "charts/accuracy_curve": wandb.plot.line(
                    wandb.Table(dataframe=log_df[["epoch", "train_acc"]]),
                    "epoch", "train_acc", title="Training Accuracy"
                )
            })
            wandb.finish()

        logger.info("=" * 70)
        logger.info("Training Completed Successfully!")
        logger.info(f"Best accuracy: {best_acc:.4f}")
        logger.info(f"Total epochs: {epoch}")
        logger.info(f"Output: {DIR}")
        logger.info("=" * 70)
    cleanup_ddp()

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPUs available")
    print(f"Starting DDP training with {num_gpus} GPUs")
    mp.spawn(main_worker, args=(num_gpus,), nprocs=num_gpus)