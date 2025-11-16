import os
import sys
import torch
import pickle
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from config import cfg
import matplotlib.pyplot as plt
from scripts.model import CHRTModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def strip_module_prefix(state):
    new_state = {}
    for k, v in state.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module."):]
        new_state[new_k] = v
    return new_state

def choose_cls_checkpoint(base_dir):
    cls_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and "cls" in p.name])
    if not cls_dirs:
        raise FileNotFoundError("CLS models not found")

    print("\nAvailable CLS models:")
    for i, d in enumerate(cls_dirs, 1):
        pt_files = list(d.glob("*.pt"))
        ckpt_name = pt_files[0].name if pt_files else "no .pt found"
        print(f"[{i}] {ckpt_name} ({d.name})")

    choice = int(input("\nSelect CLS model [number]: ")) - 1
    chosen_dir = cls_dirs[choice]
    ckpt_files = list(chosen_dir.glob("*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt file in {chosen_dir}")

    selected_ckpt = ckpt_files[0]
    print(f"\nSelected CLS model: {selected_ckpt.name}")
    return selected_ckpt

class ClsDataset(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids = ids
        self.masks = masks
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': self.ids[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx]
        }

def main():
    print("=== Start ===")
    print("Device:", cfg.device)
    print("[1] Load tokenizer...")
    if not os.path.exists(cfg.paths.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {cfg.paths.tokenizer}")
    with open(cfg.paths.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded.")
    print("  vocab_size:", tokenizer.vocab_size)
    print("  sample tokens:", tokenizer.tokens[:30])
    print("[2] Load test data...")
    if not os.path.exists(cfg.paths.cls_test):
        raise FileNotFoundError(f"test csv not found: {cfg.paths.cls_test}")
    df_test = pd.read_csv(cfg.paths.cls_test)
    print(f"Test size: {len(df_test)}")
    if 'label' not in df_test.columns or 'smiles' not in df_test.columns:
        raise ValueError("test.csv : 'smiles' and 'label'")
    print("Test label distribution:")
    print(df_test['label'].value_counts().sort_index())

    print("[3] Tokenizing test set...")
    ids, masks, labels, smiles_list = [], [], [], []
    pad_id = tokenizer.token_to_id[tokenizer.pad_token]
    for i, row in enumerate(df_test.iterrows()):
        if i % 100 == 0:
            print(f"  tokenizing row {i}/{len(df_test)}")
        _, row = row
        s = str(row['smiles'])
        seq = tokenizer.encode(s, max_length=cfg.max_len)
        mask = [1 if tok != pad_id else 0 for tok in seq]
        ids.append(seq)
        masks.append(mask)
        labels.append(int(row['label']))
        smiles_list.append(s)

    ids = torch.tensor(ids, dtype=torch.long)
    masks = torch.tensor(masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    print("[4] Creating DataLoader...")
    test_loader = DataLoader(ClsDataset(ids, masks, labels), batch_size=cfg.cls.batch_size, shuffle=False)

    print("[5] Building model...")
    model = CHRTModel(
        vocab_size=tokenizer.vocab_size,
        emb_dim=cfg.emb_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=cfg.max_len
    )
    model.classifier = model.get_classifier_head(num_classes=cfg.num_classes)

    print(f"[6] Loading model: {cfg.CLS}")
    cls_ckpt_path = choose_cls_checkpoint(cfg.paths.outputs)
    ckpt = torch.load(cls_ckpt_path, map_location=cfg.device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model_state = ckpt["model_state"]
        print("Checkpoint type: wrapped (contains model_state).")
    else:
        model_state = ckpt
        print("Checkpoint type: raw state_dict.")

    model_state = strip_module_prefix(model_state)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print("Missing keys when loading:", missing)
    print("Unexpected keys when loading:", unexpected)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total model params:", total_params)
    try:
        sample_norms = {n: p.norm().item() for n, p in model.named_parameters() if 'weight' in n and p.numel() < 200000}
        print("Some parameter norms (sample):", {k: sample_norms[k] for k in list(sample_norms)[:8]})
    except Exception:
        pass

    model.to(cfg.device)
    model.eval()

    print("[7] Start inference...")
    all_preds, all_trues = [], []
    pred_counts = [0] * cfg.num_classes

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Inference", unit="batch", file=sys.stdout)
        for batch in pbar:
            ids_b = batch['input_ids'].to(cfg.device)
            mask_b = batch['attention_mask'].to(cfg.device)
            lbl_b = batch['labels'].to(cfg.device)

            pooled = model.sequence_repr(ids_b, mask_b)
            logits = model.classifier(pooled)
            preds = logits.argmax(dim=1).cpu().tolist()

            for p in preds:
                if 0 <= p < len(pred_counts):
                    pred_counts[p] += 1

            all_preds.extend(preds)
            all_trues.extend(lbl_b.cpu().tolist())

            if len(all_preds) > 0:
                acc = accuracy_score(all_trues, all_preds)
                pbar.set_postfix({"acc": f"{acc:.4f}"})

    print("[7] Inference done.")

    print("Predicted class counts:", {i: pred_counts[i] for i in range(len(pred_counts))})
    acc = accuracy_score(all_trues, all_preds)
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(all_trues, all_preds, digits=4))

    cm = confusion_matrix(all_trues, all_preds)
    with pd.option_context('display.float_format', '{:0.2f}'.format):
        print("\nConfusion matrix (counts):")
        print(cm)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, None]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=[str(i) for i in range(cm.shape[1])],
                yticklabels=[str(i) for i in range(cm.shape[0])])
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    out_img = cfg.CONFUSION_MATRIX
    plt.savefig(out_img, dpi=150)
    print(f"Saved confusion matrix image to: {out_img}")

    mismatches = []
    for i, (t, p, s) in enumerate(zip(all_trues, all_preds, smiles_list)):
        if t != p and len(mismatches) < 12:
            mismatches.append((i, int(t), int(p), s))
    if mismatches:
        print("\nSome mismatch examples (idx, true, pred, smiles):")
        for m in mismatches:
            print(m)
    else:
        print("\nNo mismatches found")

if __name__ == "__main__":
    main()
