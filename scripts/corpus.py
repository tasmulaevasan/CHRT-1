import pandas as pd
from config import cfg

def main():
    all_smiles = []
    print(f"loading {cfg.paths.raw_train}...")
    train_df = pd.read_csv(cfg.paths.raw_train)
    print(f"loading {cfg.paths.raw_val}...")
    val_df = pd.read_csv(cfg.paths.raw_val)
    print(f"loading {cfg.paths.raw_test}...")
    test_df = pd.read_csv(cfg.paths.raw_test)

    col_name_raw = 'reactants>reagents>production'
    for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        if col_name_raw not in df.columns:
            raise ValueError(f"{name} : '{col_name_raw}'")
        all_smiles.extend(df[col_name_raw].astype(str).tolist())

    print(f"loading {cfg.paths.uspto_tpl_train}...")
    tpl_train_df = pd.read_csv(cfg.paths.uspto_tpl_train)
    print(f"loading {cfg.paths.uspto_tpl_test}...")
    tpl_test_df = pd.read_csv(cfg.paths.uspto_tpl_test)

    col_name_tpl = 'canonical_rxn'
    for df, name in [(tpl_train_df, 'tpl_train'), (tpl_test_df, 'tpl_test')]:
        if col_name_tpl not in df.columns:
            raise ValueError(f"{name} : '{col_name_tpl}'")
        all_smiles.extend(df[col_name_tpl].astype(str).tolist())

    print(f"Writing {len(all_smiles)} -> {cfg.paths.corpus}...")
    with open(cfg.paths.corpus, 'w', encoding='utf-8') as f:
        for smiles in all_smiles:
            f.write(smiles.strip() + '\n')

if __name__ == '__main__':
    main()
