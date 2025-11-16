import pandas as pd
from config import cfg
from scripts.model import mechanism_to_rxn_class
vals = set(mechanism_to_rxn_class.values())
print("unique values:", sorted(vals))

def main():
    df_train = pd.read_csv(cfg.paths.cls_train)
    df_mech = pd.read_csv(cfg.paths.uspto_31k)
    mapped = []
    for _, row in df_mech.iterrows():
        label_name = row['mechanistic_class']
        if label_name not in mechanism_to_rxn_class:
            continue
        map_val = mechanism_to_rxn_class[label_name]
        cls = int(map_val)
        smiles = row['updated_reaction']
        mapped.append({'smiles': smiles, 'label': cls})
    print(f"Valid mechanistic reactions added: {len(mapped)}")
    df_mech_mapped = pd.DataFrame(mapped)
    df_new = pd.concat([df_train, df_mech_mapped], ignore_index=True)
    print(df_new.isnull().sum())
    df_new.dropna(subset=['smiles', 'label'], inplace=True)
    print(f"Original size: {len(df_train)}, after: {len(df_new)}")
    df_new.to_csv(cfg.paths.cls_train_full, index=False)
    print(f"Saved to {cfg.paths.cls_train_full}")

if __name__ == '__main__':
    main()