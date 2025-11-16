import torch
import random
import numpy as np
from scripts.config import cfg
from torch.utils.data import Dataset

class MLMDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len=cfg.max_len, mask_prob=0.15):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

        self.pad_id = tokenizer.token_to_id[tokenizer.pad_token]
        self.mask_id = tokenizer.token_to_id[tokenizer.mask_token]
        self.unk_id = tokenizer.token_to_id[tokenizer.unk_token]

        self.vocab_start = 3
        self.vocab_end = tokenizer.vocab_size

        self.use_augmentation = cfg.augmentation.enabled
        self.aug_mode = cfg.augmentation.mode if cfg.augmentation.enabled else None
        self.aug_prob = cfg.augmentation.prob if cfg.augmentation.enabled else 0.0
        self._augmentation = None

    def _get_augmentation(self):
        if self._augmentation is None and self.use_augmentation:
            from scripts.augmentation import get_augmentation
            self._augmentation = get_augmentation(
                mode=self.aug_mode,
                prob=self.aug_prob
            )
        return self._augmentation

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        if self.use_augmentation:
            aug = self._get_augmentation()
            if aug is not None:
                smiles = aug(smiles)

        input_ids = self.tokenizer.encode(smiles, max_length=self.max_len)
        labels = [-100] * len(input_ids)

        maskable_positions = [
            i for i in range(1, len(input_ids) - 1)
            if input_ids[i] not in [self.pad_id, self.unk_id]
        ]

        if not maskable_positions:
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            attention_mask = (input_ids_tensor != self.pad_id).long()
            return {
                'input_ids': input_ids_tensor,
                'attention_mask': attention_mask,
                'labels': labels_tensor
            }

        num_to_mask = max(1, int(len(maskable_positions) * self.mask_prob))
        masked_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))

        for pos in masked_positions:
            labels[pos] = input_ids[pos]

            rand = random.random()
            if rand < 0.8:
                input_ids[pos] = self.mask_id
            elif rand < 0.9:
                input_ids[pos] = random.randint(self.vocab_start, self.vocab_end - 1)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids_tensor != self.pad_id).long()

        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask,
            'labels': labels_tensor
        }


class DynamicPaddingCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(item['input_ids'].size(0) for item in batch)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:
            seq_len = item['input_ids'].size(0)
            pad_len = max_len - seq_len

            input_ids = torch.cat([
                item['input_ids'],
                torch.full((pad_len,), self.pad_id, dtype=torch.long)
            ])

            attention_mask = torch.cat([
                item['attention_mask'],
                torch.zeros(pad_len, dtype=torch.long)
            ])

            labels = torch.cat([
                item['labels'],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list)
        }


def main():
    print(f"Loading corpus: {cfg.paths.corpus}...")
    with open(cfg.paths.corpus, 'r', encoding='utf-8') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f"Loading tokenizer: {cfg.paths.tokenizer}...")
    import pickle
    with open(cfg.paths.tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)

    dataset = MLMDataset(smiles_list, tokenizer)

    if cfg.augmentation.enabled:
        print(f"Augmentation enabled: mode={cfg.augmentation.mode}, prob={cfg.augmentation.prob}")
    else:
        print("Augmentation disabled")

    from torch.utils.data import DataLoader
    collator = DynamicPaddingCollator(pad_id=tokenizer.token_to_id[tokenizer.pad_token])
    dataloader = DataLoader(dataset, batch_size=cfg.mlm.batch_size, shuffle=True, collate_fn=collator)

    print("\n=== Testing MLM Dataset ===")
    batch = next(iter(dataloader))
    print(f"Batch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    num_masked = (batch['labels'] != -100).sum().item()
    total_tokens = batch['attention_mask'].sum().item()
    print(f"\nMasking statistics:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Masked tokens: {num_masked}")
    print(f"  Masking ratio: {num_masked / total_tokens * 100:.2f}%")

if __name__ == '__main__':
    main()