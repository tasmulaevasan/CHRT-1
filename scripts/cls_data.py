import pandas as pd
from config import cfg
from sklearn.model_selection import train_test_split

print(f"Loading DataSetB from {cfg.paths.datasetb}...")
df_b = pd.read_csv(cfg.paths.datasetb)
smiles_col = 'rxnSmiles_Mapping_NameRxn'
df_b = df_b[[smiles_col, 'rxn_Class']].rename(columns={smiles_col: 'smiles', 'rxn_Class': 'label'})
df_b['label'] -= 1

print("Splitting DataSetB into train_val (90%) and test_B (10%)...")
df_train_val, df_test_b = train_test_split(
    df_b,
    test_size=0.10,
    stratify=df_b['label'],
    random_state=42
)

print("Splitting train_val into train (80%) and val (10%)...")
df_train, df_val = train_test_split(
    df_train_val,
    test_size=0.1111,
    stratify=df_train_val['label'],
    random_state=42
)

df_a = pd.read_csv(cfg.paths.dataseta)
smiles_col_a = 'rxn_Smiles'
ndf_a = df_a[[smiles_col_a, 'rxn_Class']].rename(columns={smiles_col_a: 'smiles', 'rxn_Class': 'label'})
ndf_a['label'] -= 1

df_test = pd.concat([df_test_b, ndf_a], ignore_index=True)
print(f"Sizes:\ntrain: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

df_train.to_csv(cfg.paths.cls_train, index=False)
df_val.to_csv(cfg.paths.cls_val, index=False)
df_test.to_csv(cfg.paths.cls_test, index=False)

print(f"Saved train.csv to {cfg.paths.cls_train}")
print(f"Saved val.csv to {cfg.paths.cls_val}")
print(f"Saved test.csv to {cfg.paths.cls_test}")
