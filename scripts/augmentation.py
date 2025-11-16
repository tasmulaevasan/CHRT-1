import random

class SMILESAugmentation:
    def __init__(self, prob: float = 0.5):
        self.prob = prob
        self.use_rdkit = False
        try:
            from rdkit import Chem
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
            self.Chem = Chem
            self.use_rdkit = True
        except Exception as e:
            print(e)

    def __call__(self, smiles: str) -> str:
        if random.random() > self.prob:
            return smiles
        if self.use_rdkit:
            return self._augment_with_rdkit(smiles)
        else:
            return self._augment_fallback(smiles)

    def _augment_with_rdkit(self, smiles: str) -> str:
        try:
            mol = self.Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            augmented = self.Chem.MolToSmiles(mol, doRandom=True)
            return augmented if augmented else smiles
        except Exception:
            return smiles

    def _augment_fallback(self, smiles: str) -> str:
        if '>>' in smiles:
            parts = smiles.split('>>')
            augmented_parts = [self._rotate_string(p) for p in parts]
            return '>>'.join(augmented_parts)
        else:
            return self._rotate_string(smiles)

    def _rotate_string(self, s: str) -> str:
        if len(s) < 3:
            return s
        safe_points = []
        depth = 0
        for i, char in enumerate(s):
            if char in '([':
                depth += 1
            elif char in ')]':
                depth -= 1
            elif depth == 0 and char.isalpha():
                safe_points.append(i)

        if not safe_points:
            return s
        rotation_point = random.choice(safe_points)
        return s[rotation_point:] + s[:rotation_point]


class ReactionAugmentation:
    def __init__(self, prob: float = 0.5, shuffle_reactants: bool = True):
        self.prob = prob
        self.shuffle_reactants = shuffle_reactants
        self.smiles_aug = SMILESAugmentation(prob=1.0)

    def __call__(self, reaction_smiles: str) -> str:
        if random.random() > self.prob:
            return reaction_smiles
        if '>>' in reaction_smiles:
            parts = reaction_smiles.split('>>')
            if len(parts) != 2:
                return reaction_smiles
            reactants, products = parts
            reagents = None
        elif reaction_smiles.count('>') == 2:
            parts = reaction_smiles.split('>')
            reactants, reagents, products = parts
        else:
            return reaction_smiles

        reactants = self._augment_component_list(reactants)
        products = self._augment_component_list(products)
        if reagents is not None:
            reagents = self._augment_component_list(reagents)
            return f"{reactants}>{reagents}>{products}"
        else:
            return f"{reactants}>>{products}"

    def _augment_component_list(self, component_str: str) -> str:
        molecules = component_str.split('.')
        augmented = [self.smiles_aug(mol) for mol in molecules]
        if self.shuffle_reactants and len(augmented) > 1:
            random.shuffle(augmented)
        return '.'.join(augmented)

def get_augmentation(mode: str = 'reaction', prob: float = 0.5):
    if mode == 'reaction':
        return ReactionAugmentation(prob=prob)
    elif mode == 'smiles':
        return SMILESAugmentation(prob=prob)
    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")