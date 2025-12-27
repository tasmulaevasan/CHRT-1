import torch
import random
from torch.utils.data import Dataset

from scripts.config import cfg
from scripts.augmentation import ReactionAugmentation

class MLMDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len=cfg.max_len, mask_prob=cfg.mask_prob):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

        self.pad_id = tokenizer.token_to_id[tokenizer.pad_token]
        self.mask_id = tokenizer.token_to_id[tokenizer.mask_token]
        self.unk_id = tokenizer.token_to_id[tokenizer.unk_token]

        self.vocab_start = 3
        self.vocab_end = tokenizer.vocab_size

        self._augmentation = None
        self.augmentation = ReactionAugmentation()

    def _get_augmentation(self):
        if self._augmentation is None:
            self._augmentation = ReactionAugmentation()
        return self._augmentation

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        smiles = self.smiles_list[idx]
        if self.augmentation is not None:
            smiles = self.augmentation(smiles)

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
    import pandas as pd

    from scripts.config import cfg

    all_smiles = []
    print(f"{cfg.paths.raw_train}...")
    train_df = pd.read_csv(cfg.paths.raw_train)
    print(f"{cfg.paths.raw_val}...")
    val_df = pd.read_csv(cfg.paths.raw_val)
    print(f"{cfg.paths.raw_test}...")
    test_df = pd.read_csv(cfg.paths.raw_test)

    col_name_raw = 'reactants>reagents>production'
    for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        if col_name_raw not in df.columns:
            raise ValueError(f"{name} : '{col_name_raw}'")
        all_smiles.extend(df[col_name_raw].astype(str).tolist())

    print(f"{cfg.paths.uspto_tpl_train}...")
    tpl_train_df = pd.read_csv(cfg.paths.uspto_tpl_train)
    print(f"{cfg.paths.uspto_tpl_test}...")
    tpl_test_df = pd.read_csv(cfg.paths.uspto_tpl_test)

    col_name_tpl = 'canonical_rxn'
    for df, name in [(tpl_train_df, 'tpl_train'), (tpl_test_df, 'tpl_test')]:
        if col_name_tpl not in df.columns:
            raise ValueError(f"{name} : '{col_name_tpl}'")
        all_smiles.extend(df[col_name_tpl].astype(str).tolist())

    print(f"{len(all_smiles)} -> {cfg.paths.corpus}...")
    with open(cfg.paths.corpus, 'w', encoding='utf-8') as f:
        for smiles in all_smiles:
            f.write(smiles.strip() + '\n')

if __name__ == "__main__":
    main()