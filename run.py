from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
import random

import argparse
import numpy as np
import pandas as pd
from generate import get_penalty, logP_calculator


target_logP_low = 2
target_logP_high = 3

count = 0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path", default="./check.csv", type=str, help="input path"
)
parser.add_argument(
    "--output_path", default="./output.csv", type=str, help="output path"
)
parser.add_argument("--iters", default=50, type=int, help="iteraions")
args = parser.parse_args()

df = pd.read_csv(args.input_path, header=None)

def generate_molecule(target_logP_low, target_logP_high, df_mol):
    # Инициализация случайной молекулы
    mol = Chem.MolFromSmiles(df_mol)
    mol = Chem.AddHs(mol)
    min_penalty = None
    min_smiles = None
    min_mol = None
    min_logP = None

    for i in range(50):
        new_mol = Chem.Mol(mol)
        AllChem.EmbedMolecule(new_mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)

        if new_mol.GetNumConformers() > 0:
            AllChem.UFFOptimizeMolecule(new_mol)
            smiles = Chem.MolToSmiles(new_mol)
            logP = logP_calculator(smiles)

            if not min_penalty or get_penalty(logP) < min_penalty:
                min_penalty = get_penalty(logP)
                min_smiles = smiles
                min_mol = new_mol
                min_logP = logP
        mol = new_mol
    return min_mol, min_logP, min_smiles

smiles_result = []
for i in df[0].tolist():
    generated_molecule, logP, smiles = generate_molecule(target_logP_low, target_logP_high, i)
    #print(f"Сгенерированная молекула: {smiles}, logP: {logP}")
    smiles_result.append(smiles)
pd.DataFrame(smiles_result, columns=['output']).to_csv(args.output_path, index=None, header=None)