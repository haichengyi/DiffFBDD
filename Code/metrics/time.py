import csv

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolAlign, rdFMCS, QED, Crippen, Lipinski
from analysis.SA_Score.sascorer import calculateScore

import pandas as pd
import os

path_new = r'/home/linux/Downloads/process_result/result_121_processed_2/pocket_times' #ligand = node_ligand/10
file_new = []
for root, dirs, files in os.walk(path_new):
    for file in files:
        path = os.path.join(root, file)
        file_new.append(path)
print(len(file_new))
sum = 0
length = 0
for i in file_new:
    with open(i) as f:
        content = f.read()
        sum += float(content.split(' ')[1])
        length = length + 1
print(sum/length)

