import random
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit import RDLogger
from joblib import load


RDLogger.DisableLog("rdApp.*")
MODEL_LOGP = load("logP_model.joblib")


def ecfc_molstring(molecule, radius=3, size=1024):
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetHashedMorganFingerprint(molecule, radius, size, useFeatures=False),
        arr,
    )
    return arr


def logP_calculator(smiles: str) -> float:
    """

    :param smiles: smiles string like "c1ccccc1"
    :return: logP value
    """
    features = np.array([ecfc_molstring(Chem.MolFromSmiles(smiles))])
    return MODEL_LOGP.predict(features)


def mutate_population(
    top_worst_mols: [str], top_best_mols: [str], num_iterations=50
) -> str:
    """
    Split 'good' and 'bad' molecules in fragments and replace one fragment in 'bad' molecule with fragment from 'good'
    molecule, make population of such mutant molecules and choose best of them
    :param top_worst_mols: best molecules from population by logP value
    :param top_best_mols: the best molecules from population by logP value
    :param num_iterations: amount of single exchanges between 'good' and 'bad' fragments
    :return: the best mutant molecule smiles string
    """
    raw_generation = []
    for mol in top_worst_mols:
        replacement_genes_bank = []
        for i in top_best_mols:
            replacement_genes_bank.extend(
                get_branches_from_smiles(
                    Chem.MolToSmiles(
                        Chem.MolFromSmiles(i), canonical=False, isomericSmiles=False
                    )
                )
            )
        mol = Chem.MolToSmiles(
            Chem.MolFromSmiles(mol), canonical=False, isomericSmiles=False
        )
        target_mol_genes = get_branches_from_smiles(mol)
        for _ in range(num_iterations):
            try:
                raw_generation.append(
                    mol.replace(
                        random.choice(target_mol_genes),
                        random.choice(replacement_genes_bank),
                        1,
                    )
                )
            except:
                pass
    raw_generation = [i for i in raw_generation if Chem.MolFromSmiles(i)]
    return sorted(raw_generation, key=lambda x: logP_calculator(x))[0]


def get_penalty(y_pred, min_target=2.0, max_target=3.0):
    if y_pred > max_target:
        return abs(y_pred - max_target)
    elif y_pred < min_target:
        return abs(y_pred - min_target)
    else:
        return 0


def search_step(smiles_pop: [str], top_to_change=5) -> [str]:
    smiles_pop = sorted(smiles_pop, key=lambda x: get_penalty(logP_calculator(x)))
    top_worst_mols = smiles_pop[-top_to_change:]
    top_best_mols = smiles_pop[:top_to_change]

    new_mol = mutate_population(top_worst_mols, top_best_mols)

    smiles_pop.pop(-1)
    smiles_pop.append(new_mol)

    return smiles_pop


def get_branches_from_smiles(smiles: str) -> [str]:
    """
    Get all possible branches from smiles string
    :param smiles:
    :return:
    """
    branches = re.findall(r"(\(.*?\))", smiles)
    for branch in branches:
        bracket_diff = branch.count("(") - branch.count(")")
        if bracket_diff > 0:
            branches[branches.index(branch)] = branch + ")" * bracket_diff
        else:
            branches[branches.index(branch)] = "(" * bracket_diff + branch
    return branches
