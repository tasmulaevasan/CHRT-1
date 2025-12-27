import os
import torch
import random
import numpy as np
from pathlib import Path
import torch.distributed as dist

from scripts.config import cfg
from scripts.model import CHRTModel

def setup_env():
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)

def setup_ddp(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def choose_mlm_checkpoint(base_dir: Path) -> Path:
    mlm_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and "mlm" in p.name])
    if not mlm_dirs:
        raise FileNotFoundError("MLM models not found")

    print("Available MLM models:")
    for i, d in enumerate(mlm_dirs, 1):
        pt_files = list(d.glob("*.pt"))
        ckpt = pt_files[0].name if pt_files else "no .pt found"
        print(f"[{i}] {ckpt}")

    choice = 0
    chosen_dir = mlm_dirs[choice]
    ckpt_files = list(chosen_dir.glob("*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt file in {chosen_dir}")
    return ckpt_files[0]

def choose_cls_checkpoint(base_dir: Path) -> Path:
    cls_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and "cls" in p.name])
    if not cls_dirs:
        raise FileNotFoundError(f"No CLS models found in {base_dir}")

    print("=" * 70)
    print("Available CLS models:")
    for i, d in enumerate(cls_dirs, 1):
        pt_files = list(d.glob("*_best.pt"))
        if not pt_files:
            pt_files = list(d.glob("*.pt"))
        ckpt_name = pt_files[0].name if pt_files else "no .pt found"
        print(f"[{i}] {ckpt_name} ({d.name})")

    choice = int(input("Select CLS model [number]: ")) - 1

    if choice < 0 or choice >= len(cls_dirs):
        raise ValueError(f"Invalid choice: {choice + 1}")

    chosen_dir = cls_dirs[choice]

    ckpt_files = list(chosen_dir.glob("*_best.pt"))
    if not ckpt_files:
        ckpt_files = list(chosen_dir.glob("*_final.pt"))
    if not ckpt_files:
        ckpt_files = list(chosen_dir.glob("*.pt"))

    if not ckpt_files:
        raise FileNotFoundError(f"No .pt file in {chosen_dir}")

    selected_ckpt = ckpt_files[0]
    print(f"Selected: {selected_ckpt.name}")
    print("=" * 70)
    return selected_ckpt

def strip_module_prefix(state):
    new_state = {}
    for k, v in state.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        new_state[new_k] = v
    return new_state

def load_model(checkpoint_path: Path, tokenizer, device, logger):
    logger.info(f"Building model...")
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
    )

    logger.info(f"Loading checkpoint: {checkpoint_path.name}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model_state = ckpt["model_state"]
        logger.info("Checkpoint metadata:")
        if "epoch" in ckpt:
            logger.info(f"  Epoch: {ckpt['epoch']}")
        if "val_acc" in ckpt:
            logger.info(f"  Val Acc: {ckpt['val_acc']:.4f}")
        if "val_f1" in ckpt:
            logger.info(f"  Val F1: {ckpt['val_f1']:.4f}")
    else:
        model_state = ckpt
        logger.info("Checkpoint type: raw state_dict")

    model_state = strip_module_prefix(model_state)

    missing, unexpected = model.load_state_dict(model_state, strict=False)

    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
        if len(missing) <= 10:
            for k in missing:
                logger.warning(f"  - {k}")

    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 10:
            for k in unexpected:
                logger.warning(f"  - {k}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    model.to(device)
    model.eval()
    return model

def save_results(logger, metrics, output_dir: Path):
    results_file = output_dir / "results.txt"

    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n")

        f.write(f"Overall Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)\n")
        f.write(f"Macro F1 Score:       {metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted F1 Score:    {metrics['f1_weighted']:.4f}\n")
        f.write(f"Macro Precision:      {metrics['precision']:.4f}\n")
        f.write(f"Macro Recall:         {metrics['recall']:.4f}\n")
        f.write("=" * 70 + "\n")
    logger.info(f"Results: {results_file}")