import torch
import pickle
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from scripts.config import cfg
from scripts.cls_data import ClsDataset, SimpleCollator
from scripts.logger import get_logger, global_exception_logger
from scripts.functions import choose_cls_checkpoint, load_model, save_results

def evaluate(model, dataloader, device, logger):
    logger.info("Starting evaluation...")

    all_preds = []
    all_trues = []
    all_logits = []

    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", ncols=100)
        for batch in pbar:
            ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels']

            pooled = model.sequence_repr(ids, mask)
            logits = model.classifier(pooled)

            preds = logits.argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_trues.extend(labels.tolist())
            all_logits.append(logits.cpu())

            if len(all_preds) > 0:
                acc = accuracy_score(all_trues, all_preds)
                pbar.set_postfix({"acc": f"{acc:.4f}"})

    all_logits = torch.cat(all_logits)
    return all_preds, all_trues, all_logits

def compute_metrics(y_true, y_pred, class_names, logger):
    logger.info("=" * 70)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 70)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    logger.info(f"Overall Accuracy:     {acc:.4f} ({acc * 100:.2f}%)")
    logger.info(f"Macro F1 Score:       {f1_macro:.4f}")
    logger.info(f"Weighted F1 Score:    {f1_weighted:.4f}")
    logger.info(f"Macro Precision:      {precision:.4f}")
    logger.info(f"Macro Recall:         {recall:.4f}")
    logger.info("=" * 70)

    logger.info("Per-Class Classification Report:")
    report = classification_report(
        y_true, y_pred,
        digits=4,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    logger.info(classification_report(
        y_true, y_pred,
        digits=4,
        target_names=class_names,
        zero_division=0
    ))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "report": report
    }


def plot_confusion_matrix(logger, y_true, y_pred, output_dir: Path, class_names=None):
    logger.info("Generating confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, None] + 1e-9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names if class_names else range(cm.shape[1]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
        ax=ax1
    )
    ax1.set_xlabel("Predicted Class")
    ax1.set_ylabel("True Class")
    ax1.set_title("Confusion Matrix (Counts)")

    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names if class_names else range(cm.shape[1]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
        ax=ax2,
        vmin=0, vmax=1
    )
    ax2.set_xlabel("Predicted Class")
    ax2.set_ylabel("True Class")
    ax2.set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()

    # Save
    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved confusion matrix: {output_path}")

    plt.close()

    return cm, cm_norm


def analyze_errors(logger, y_true, y_pred, test_df, output_dir: Path, top_k=20):
    logger.info("=" * 70)
    logger.info("ERROR ANALYSIS")
    logger.info("=" * 70)

    mismatches = []
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label != pred_label:
            smiles = test_df.iloc[i]['smiles'] if 'smiles' in test_df.columns else "N/A"
            mismatches.append({
                'idx': i,
                'true_label': true_label,
                'pred_label': pred_label,
                'smiles': smiles
            })

    error_rate = len(mismatches) / len(y_true)
    logger.info(f"Total misclassifications: {len(mismatches)} / {len(y_true)} ({error_rate * 100:.2f}%)")

    if mismatches:
        error_df = pd.DataFrame(mismatches)
        error_csv = output_dir / "misclassified_samples.csv"
        error_df.to_csv(error_csv, index=False)
        logger.info(f"Saved misclassifications: {error_csv}")

        logger.info(f"Top {min(top_k, len(mismatches))} misclassifications:")
        for i, err in enumerate(mismatches[:top_k], 1):
            logger.info(f"[{i}] True: {err['true_label']} | Pred: {err['pred_label']}")
            logger.info(f"SMILES: {err['smiles'][:80]}...")
    else:
        logger.info("No misclassifications found")

    logger.info("=" * 70)

def main():
    checkpoint_path = choose_cls_checkpoint(cfg.paths.outputs)

    output_dir = checkpoint_path.parent / "evaluation"
    output_dir.mkdir(exist_ok=True)

    log_file = output_dir / "evaluation.log"
    logger = get_logger(__name__, log_file=str(log_file))
    global_exception_logger(logger)

    logger.info("=" * 70)
    logger.info("CLS EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Random seed: {cfg.seed}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")

    logger.info(f"[1/6] Loading tokenizer from {cfg.paths.tokenizer}...")
    if not cfg.paths.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found: {cfg.paths.tokenizer}")

    with open(cfg.paths.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)

    logger.info(f"Vocabulary size: {tokenizer.vocab_size:,}")

    logger.info(f"[2/6] Loading test data from {cfg.paths.cls_test}...")
    if not cfg.paths.cls_test.exists():
        raise FileNotFoundError(f"Test file not found: {cfg.paths.cls_test}")

    df_test = pd.read_csv(cfg.paths.cls_test)
    logger.info(f"Test set size: {len(df_test):,} samples")

    if 'label' not in df_test.columns or 'smiles' not in df_test.columns:
        raise ValueError("Test CSV must contain 'smiles' and 'label' columns")

    logger.info("Test set class distribution:")
    class_counts = df_test['label'].value_counts().sort_index()
    for cls, count in class_counts.items():
        pct = count / len(df_test) * 100
        logger.info(f"  Class {cls}: {count:>5,} ({pct:>5.2f}%)")

    logger.info(f"[3/6] Creating test dataset...")
    test_dataset = ClsDataset(df_test, tokenizer)
    collator = SimpleCollator(pad_id=tokenizer.token_to_id[tokenizer.pad_token])

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.cls.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    logger.info(f"Test batches: {len(test_loader):,}")

    logger.info("[4/6] Loading model...")
    device = torch.device(cfg.device)
    model = load_model(checkpoint_path, tokenizer, device, logger)

    logger.info("[5/6] Running evaluation...")
    y_pred, y_true, logits = evaluate(model, test_loader, device, logger)

    logger.info("[6/6] Computing metrics...")
    class_names = [f"Class_{i}" for i in range(cfg.num_classes)]
    metrics = compute_metrics(y_true, y_pred, class_names, logger)

    plot_confusion_matrix(logger, y_true, y_pred, output_dir, class_names)
    analyze_errors(logger, y_true, y_pred, df_test, output_dir)
    save_results(logger, metrics, output_dir)

    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETED")
    logger.info(f"Results: {output_dir}")
    logger.info(f"Logs:    {log_file}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()