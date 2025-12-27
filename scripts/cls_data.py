import torch
from torch.utils.data import Dataset

from scripts.config import cfg

class SimpleCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, samples):
        max_len = max(len(s['input_ids']) for s in samples)
        ids, mask, labels = [], [], []
        for s in samples:
            seq = s['input_ids']
            pad_n = max_len - len(seq)
            ids.append(seq + [self.pad_id] * pad_n)
            mask.append([1] * len(seq) + [0] * pad_n)
            labels.append(s['labels'])
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

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

def main():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    mechanism_to_rxn_class = {
        "DCC_condensation": 1,
        "carboxylic_acid_derivative_hydrolysis_or_formation": 1,
        "methyl_ester_synthesis": 1,
        "Weinreb_ketone_synthesis": 1,
        "base_cat_ester_hydrolysis": 1,
        "intramolecular_lactonization": 1,
        "Grignard": 2,
        "alkynyl_attack_to_carbonyl": 2,
        "Hantzsch_thiazole_synthesis": 3,
        "imidazole_synthesis": 3,
        "Knorr_pyrazole_synthesis": 3,
        "Paal_Knorr_pyrrole_synthesis": 3,
        "Cbz_deprotection": 5,
        "Boc_deprotection": 5,
        "Fmoc_deprotection": 5,
        "O_demethylation": 5,
        "nucleophilic_attack_to_(thio)carbonyl_or_sulfonyl": 8,
        "alcohol_attack_to_carbonyl_or_sulfonyl": 8,
        "nucleophilic_attack_to_iso(thio)cyanate": 8,
        "isothiocyanate_synthesis": 8,
        "alkene_epoxidation": 9,
        "Markovnikov_addition": 9,
        "acetal_formation": 9,
        "acetal_formation_from_enol_ether": 9,
        "(hemi)acetal(aminal)_hydrolysis": 9,
        "SN1": 10,
        "SN2": 10,
        "SN2_alcohol(thiol)": 10,
        "SN2_with_tosylate": 10,
        "SN1_with_tosylate": 10,
        "Mitsunobu": 10,
        "double_SN2": 10,
        "SNAr(ortho)": 11,
        "SNAr(para)": 11,
        "SNAr_alco(thi)ol(para)": 11,
        "SNAr_alco(thi)ol(ortho)": 11,
        "aldol_condensation": 12,
        "aldol_addition": 12,
        "Mannich": 12,
        "Wittig": 13,
        "Wittig_ver_2": 13,
        "Horner_Wadsworth_Emmons": 13,
        "reductive_amination": 14,
        "imine_formation": 14,
        "imine_reduction": 14,
        "carbonyl_reduction": 15,
        "ester_reduction": 15,
        "nitrile_reduction": 15,
        "amide_reduction": 15,
        "lactone_reduction": 15,
        "sulfide_oxidation": 16,
        "sulfide_oxidation_by_peroxide": 16,
        "Jones_oxidation": 16,
        "Swern_oxidation": 16,
        "amine_oxidation": 16,
        "Wolf_Kishner_reduction": 17,
        "Staudinger": 17,
        "Appel": 17,
        "Ing_Manske": 17,
        "Vilsmeier_formylation": 17,
        "Friedel_Crafts_acylation": 17,
        "primary_amide_dehydration": 17,
    }

    CLASS_NAMES = {
        0: "Алкилирование и арилирование гетероатомов",
        1: "Ацилирование и гидролиз",
        2: "Образование связей C-C",
        3: "Образование гетероциклов",
        4: "Защита функциональных групп",
        5: "Снятие защиты функциональных групп",
        6: "Восстановление (общее)",
        7: "Окисление (общее)",
        8: "Нуклеофильные преобразования функциональных групп",
        9: "Добавление к двойным связям",
        10: "Нуклеофильное замещение (SN)",
        11: "Ароматическое замещение (SNAr)",
        12: "Альдольные реакции",
        13: "Реакции Виттига",
        14: "Восстановительное аминирование",
        15: "Специализированное восстановление",
        16: "Специализированное окисление",
        17: "Специальные реакции",
    }

    print("=" * 70)
    print("CLS DATASET")
    print("=" * 70)

    print(f"[1/6] Loading DatasetB from {cfg.paths.datasetb}...")
    df_b = pd.read_csv(cfg.paths.datasetb)
    smiles_col = 'rxnSmiles_Mapping_NameRxn'
    df_b = df_b[[smiles_col, 'rxn_Class']].rename(
        columns={smiles_col: 'smiles', 'rxn_Class': 'label'}
    )
    df_b['label'] -= 1
    print(f"Loaded: {len(df_b):,} reactions")

    print(f"[2/6] Loading and mapping USPTO-31k from {cfg.paths.uspto_31k}...")
    df_mech = pd.read_csv(cfg.paths.uspto_31k)

    mapped = []
    skipped = []

    for _, row in df_mech.iterrows():
        label_name = row['mechanistic_class']

        if label_name not in mechanism_to_rxn_class:
            skipped.append(label_name)
            continue

        cls = int(mechanism_to_rxn_class[label_name])
        smiles = row['updated_reaction']
        mapped.append({'smiles': smiles, 'label': cls})

    print(f"Mapped: {len(mapped):,} reactions")
    if skipped:
        print(f"Skipped: {len(skipped):,} reactions ({len(set(skipped))} unique mechanisms)")

    df_mech_mapped = pd.DataFrame(mapped)

    print(f"[3/6] Combining DatasetB + USPTO-31k...")
    df_combined = pd.concat([df_b, df_mech_mapped], ignore_index=True)
    df_combined.dropna(subset=['smiles', 'label'], inplace=True)
    print(f"Combined: {len(df_combined):,} reactions")
    print(f"Classes present: {sorted(df_combined['label'].unique())}")

    print("[4/6] Creating stratified splits on COMBINED data...")

    df_train_val, df_test = train_test_split(
        df_combined,
        test_size=0.10,
        stratify=df_combined['label'],
        random_state=42
    )

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=0.1111,
        stratify=df_train_val['label'],
        random_state=42
    )

    print(f"Train: {len(df_train):,} reactions")
    print(f"Val:   {len(df_val):,} reactions")
    print(f"Test:  {len(df_test):,} reactions")

    print("[5/6] Adding DatasetA to test set...")
    df_a = pd.read_csv(cfg.paths.dataseta)
    smiles_col_a = 'rxn_Smiles'
    ndf_a = df_a[[smiles_col_a, 'rxn_Class']].rename(
        columns={smiles_col_a: 'smiles', 'rxn_Class': 'label'}
    )
    ndf_a['label'] -= 1
    print(f"DatasetA: {len(ndf_a):,} reactions")

    df_test = pd.concat([df_test, ndf_a], ignore_index=True)
    print(f"Test (with DatasetA): {len(df_test):,} reactions")

    print("Sizes:")
    print(f"Train: {len(df_train):,} reactions")
    print(f"Val:   {len(df_val):,} reactions")
    print(f"Test:  {len(df_test):,} reactions")
    print(f"Total: {len(df_train) + len(df_val) + len(df_test):,} reactions")

    train_classes = set(df_train['label'].unique())
    val_classes = set(df_val['label'].unique())
    test_classes = set(df_test['label'].unique())

    print(f"Train classes: {sorted(train_classes)}")
    print(f"Val classes:   {sorted(val_classes)}")
    print(f"Test classes:  {sorted(test_classes)}")

    missing_in_val = train_classes - val_classes
    missing_in_test = train_classes - test_classes

    if missing_in_val:
        print(f"WARNING: Classes missing in VAL: {sorted(missing_in_val)}")
    if missing_in_test:
        print(f"WARNING: Classes missing in TEST: {sorted(missing_in_test)}")

    print(f"{'Set':<8} {'Class':<7} {'Name':<45} {'Count':>8} {'%':>6}")

    for split_name, split_df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        class_counts = split_df['label'].value_counts().sort_index()
        total = len(split_df)

        for cls in range(18):
            count = class_counts.get(cls, 0)
            pct = (count / total * 100) if total > 0 else 0
            name = CLASS_NAMES.get(cls, "Unknown")
            if count > 0:
                print(f"{split_name:<8} {cls:<7} {name:<45} {count:>8,} {pct:>5.1f}%")

    train_class_counts = df_train['label'].value_counts().sort_index()
    train_total = len(df_train)

    print(f"{'Class':<7} {'Name':<45} {'Count':>8} {'%':>6}")

    for cls in range(18):
        count = train_class_counts.get(cls, 0)
        pct = (count / train_total * 100) if train_total > 0 else 0
        name = CLASS_NAMES.get(cls, "Unknown")
        print(f"{cls:<7} {name:<45} {count:>8,} {pct:>5.1f}%")

    min_class = train_class_counts.min()
    max_class = train_class_counts.max()
    print(f"Statistics:")
    print(f"Min size: {min_class:,}")
    print(f"Max size: {max_class:,}")
    print(f"Avg size: {train_total // 18:,}")
    print(f"Imbalance ratio: {max_class / min_class:.1f}x")

    df_train.to_csv(cfg.paths.cls_train, index=False)
    df_val.to_csv(cfg.paths.cls_val, index=False)
    df_test.to_csv(cfg.paths.cls_test, index=False)

    print(f"Saved: {cfg.paths.cls_train}")
    print(f"Saved: {cfg.paths.cls_val}")
    print(f"Saved: {cfg.paths.cls_test}")

if __name__ == "__main__":
    main()