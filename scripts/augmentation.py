import random

class BaseAugmentation:
    def __init__(self):
        self.use_rdkit = False
        try:
            from rdkit import Chem
            self.use_rdkit = True
        except Exception as e:
            print(f"RDKit not available: {e}")
            self.use_rdkit = False

    def _get_chem(self):
        if self.use_rdkit:
            try:
                from rdkit import Chem
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')
                return Chem
            except:
                return None
        return None

class SMILESAugmentation(BaseAugmentation):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, smiles: str) -> str:
        if random.random() > self.prob:
            return smiles
        if self.use_rdkit:
            return self._augment_with_rdkit(smiles)
        else:
            return self._augment_fallback(smiles)

    def _augment_with_rdkit(self, smiles: str) -> str:
        Chem = self._get_chem()
        if Chem is None:
            return smiles

        try:
            if '>>' in smiles or '>' in smiles:
                if '>>' in smiles:
                    parts = smiles.split('>>')
                else:
                    parts = smiles.split('>')

                augmented_parts = []
                for part in parts:
                    if not part:
                        augmented_parts.append('')
                        continue
                    molecules = part.split('.')
                    aug_mols = []
                    for mol_smi in molecules:
                        mol = Chem.MolFromSmiles(mol_smi)
                        if mol is None:
                            aug_mols.append(mol_smi)
                        else:
                            aug = Chem.MolToSmiles(mol, doRandom=True)
                            aug_mols.append(aug if aug else mol_smi)
                    augmented_parts.append('.'.join(aug_mols))

                separator = '>>' if '>>' in smiles else '>'
                return separator.join(augmented_parts)
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return smiles
                augmented = Chem.MolToSmiles(mol, doRandom=True)
                return augmented if augmented else smiles
        except Exception:
            return smiles

    def _augment_fallback(self, smiles: str) -> str:
        if '>>' in smiles or '>' in smiles:
            return smiles

        if len(smiles) < 3:
            return smiles
        rotation_point = random.randint(0, len(smiles) - 1)
        return smiles[rotation_point:] + smiles[:rotation_point]

class ReactionAugmentation:
    def __init__(self, prob: float = 0.5, shuffle_reactants: bool = True):
        self.prob = prob
        self.shuffle_reactants = shuffle_reactants
        self.smiles_aug = SMILESAugmentation(prob=1.0)

    def __call__(self, reaction_smiles: str) -> str:
        if random.random() > self.prob:
            return reaction_smiles

        if '>>' in reaction_smiles:
            reactants, products = reaction_smiles.split('>>')
            reagents = None
        elif reaction_smiles.count('>') == 2:
            reactants, reagents, products = reaction_smiles.split('>')
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