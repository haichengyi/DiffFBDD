import csv
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolAlign, rdFMCS, QED, Crippen, Lipinski
from analysis.SA_Score.sascorer import calculateScore
import pandas as pd
import os

def similarity(mol_a, mol_b):
    fp1 = Chem.RDKFingerprint(mol_a)
    fp2 = Chem.RDKFingerprint(mol_b)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def rules(pocket_mols):
    rule_1 = Descriptors.ExactMolWt(pocket_mols) < 500
    rule_2 = Lipinski.NumHDonors(pocket_mols) <= 5
    rule_3 = Lipinski.NumHAcceptors(pocket_mols) <= 10
    rule_4 = (logp := Crippen.MolLogP(pocket_mols) >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(pocket_mols) <= 10
    mean_rule = np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
    return mean_rule

def sim(pocket_mols,original_mol):
    fp1 = Chem.RDKFingerprint(pocket_mols)
    fp2 = Chem.RDKFingerprint(original_mol)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity

def div(mol):
    # # diversity
    if len(mols) < 2:
        diversity = 0.0
    div = 0
    total = 0
    for sss in range(len(mols)):
        for ttt in range(sss + 1, len(mols)):
            div += 1 - similarity(mols[sss], mols[ttt])
            total += 1
    diversity = div / total
    return diversity

# Then, we can create an empty dataframe with the desired column names:
results_df = pd.DataFrame(columns=['分子量','num_atom', 'LogP','QED', 'SA', 'Lipinski', 'Similarity','Diversity','Vina_score','ligand'])
path_ori = r'/home/linux/Documents/DiffSBDD-Data/data_pre/processed_crossdock_noH_full_temp/test'
path_new = r'/home/linux/PycharmProjects/out_dir_ca_full_apart/r531_2_no_part/processed'
path_new = r''
# path_new = '/home/linux/myself_run/log/yihaicheng/517_3/processed'  #full
pocket = r'/home/linux/Documents/DiffSBDD-Data/processed_crossdock_noH_full_temp/test'
file_new = []
file_ori = []
file_pocket = []
for root, dirs, files in os.walk(path_new):
    for file in files:
        path = os.path.join(root, file)
        file_new.append(path)
print(len(file_new))
for root, dirs, files in os.walk(path_ori):
    for file in files:
        path = os.path.join(root, file)
        if 'sdf' in file:
            file_ori.append(path)
print(len(file_ori))
for root, dirs, files in os.walk(pocket):
    for file in files:
        path = os.path.join(root, file)
        if 'pdb' in file:
            file_pocket.append(path)
print(len(file_pocket))
# qvina
or_l = pd.read_csv('/home/linux/R516/ori_vina/qvina2_scores.csv')#crossDock
df = pd.read_csv('/home/linux/PycharmProjects/Quvina/Node_allow/qvina2_scores.csv') #Ligand_node_allow
# df = pd.read_csv('/home/linux/PycharmProjects/out_dir_ca_full_apart/qvina_crossdocked/q531_2_no_part/qvina2_scores.csv') #ligand = ligand_node/10
# df = pd.read_csv('/home/linux/r517/qvina2_scores.csv')  #full
# df = pd.read_csv('/home/linux/r517/q_ca/qvina2_scores.csv')  #ca
temp_full = pd.read_csv('/home/linux/PycharmProjects/DiffSBDD-main-before/DiffSBDD-main/result725_full.csv')
names = temp_full["ligand"]
names = list(set(names))
for ori in file_ori:
    for new in file_new:
        resultsdd = {}
        if ori.split('/')[-1].split('.')[0] in new.split('/')[-1].split('.')[0]:
            pocket_mols_sdf = new
            len1 = len(Chem.SDMolSupplier(pocket_mols_sdf))
            mols = Chem.SDMolSupplier(pocket_mols_sdf)
            for ss in range(len(df['ligand'])):
                if str(df['ligand'][ss]).split('/')[-1] == new.split('/')[-1]:
                    vina = str(df['scores'][ss]).split('[')[1]
            for tt in range(len(or_l['ligand'])):
                if str(or_l['ligand'][tt]).split('/')[-1] == ori.split('/')[-1]:
                    vina_ori = or_l['scores'][tt]

            for i in range(len(Chem.SDMolSupplier(pocket_mols_sdf))):
                pocket_mols = Chem.SDMolSupplier(pocket_mols_sdf)[i]
                if pocket_mols:
                    Chem.SanitizeMol(pocket_mols)
                    # print(pocket_mols.GetNumAtoms())
                original_mol_sdf = ori
                original_mol = Chem.SDMolSupplier(original_mol_sdf)[0]
                try:
                    iteration_result1 = {'分子量': Descriptors.MolWt(pocket_mols),

                                         'num_atom':pocket_mols.GetNumAtoms(),
                                         #
                                         'LogP': Descriptors.MolLogP(pocket_mols),
                                         #
                                         'QED': QED.qed(pocket_mols),
                                         #
                                         'SA': round((10 - calculateScore(pocket_mols)) / 9, 2),
                                         #
                                         'Lipinski': rules(original_mol),
                                         #
                                         'Similarity': sim(pocket_mols,original_mol),
                                         #
                                         'Diversity': div(mols),
                                         #
                                         # # 'Vina_score': str(vina),
                                         'Vina_score': float(vina.split(',')[i].split(']')[0]),
                                         # #
                                         'ligand':new.split('/')[-1]
                                         }

                    iteration_result2 = {
                        '分子量': Descriptors.MolWt(original_mol),

                        'num_atom':original_mol.GetNumAtoms(),

                        'LogP': Descriptors.MolLogP(original_mol),

                        'QED': QED.qed(original_mol),

                        'SA': round((10 - calculateScore(original_mol)) / 9, 2),

                        'Lipinski': rules(original_mol),

                        'Similarity': sim(original_mol,original_mol),

                        'Diversity': 0,

                        'Vina_score': str(vina_ori).split('[')[1].split(']')[0],

                        'ligand': ori.split('/')[-1]
                    }
                    # if (iteration_result1['num_atom']) - iteration_result2['num_atom'] > -10  and iteration_result1['Vina_score'] < -5 :
                    results_df = results_df.append(iteration_result1, ignore_index=True)
                        # results_df = results_df.append(iteration_result2, ignore_index=True)

                except:
                    i = i+1
                    continue
csv_path = './result825.csv'
results_df.to_csv(csv_path, index=False)
# result = []
# for i in results_df:
#     if i == 'ligand':
#         continue
#     else:
#         result.append(float(results_df[i].sum(axis=0)) / len(results_df['ligand']))
# mean = {'分子量': result[0],'num_atom':result[1],'LogP': result[2],
#         'QED': result[3],'SA': result[4],'Lipinski': result[5],
#         'Similarity': result[6],'Diversity': result[7],
#         'Vina_score': result[8],'ligand': 'ALL'}
# print(mean)
# results_df = results_df.append(mean, ignore_index=True)
# results_df.to_csv(csv_path, index=False)