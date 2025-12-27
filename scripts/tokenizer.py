import os
import pickle

from scripts.config import cfg
from scripts.model import SmilesTokenizer

def main():
    print(f"Loading corpus : {cfg.paths.corpus}...")
    with open(cfg.paths.corpus, 'r', encoding='utf-8') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print("Building tokenizer...")
    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(smiles_list)
    print(f"Saving tokenizer -> {cfg.paths.tokenizer}...")
    os.makedirs(os.path.dirname(cfg.paths.tokenizer), exist_ok=True)
    with open(cfg.paths.tokenizer, 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Tokenizer saved successfully")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Sample tokens (first 30): {tokenizer.tokens[:30]}")

if __name__ == '__main__':
    main()