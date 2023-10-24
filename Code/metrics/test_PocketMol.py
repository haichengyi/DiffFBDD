import csv
from datetime import datetime

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
def read_tsv(path):
    data = pd.read_csv(path,sep='\t')
    return data.iloc[:,1]
def con_mean(sum):
    average_dict = {}
    for i in sum:
        for key,value in i.items():
            if key in average_dict and key != 'ligand':
                average_dict[key] += value
            else:
                average_dict[key] = value
    num_dicts = len(sum)
    for key in average_dict.keys():
        if key == 'ligand':
            continue
        average_dict[key] /= num_dicts

    return average_dict

def read_txt(path):
    print(path)
    with open(path, "r", encoding="gbk") as f:
        r = f.readlines()
        time_start = r[0].split(',')[0].split('[')[1]
        time_start = datetime.strptime(time_start,"%Y-%m-%d %H:%M:%S")
        time_end = r[-1].split(',')[0].split('[')[1]
        time_end = datetime.strptime(time_end,"%Y-%m-%d %H:%M:%S")
        time = time_end - time_start
        sum_time = t2s(time)
        t = float(r[-1].split('|')[1].split(' ')[2])
        return sum_time/t
def t2s(t):
    t = str(t)
    h,m,s = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)

# Then, we can create an empty dataframe with the desired column names:
results_df = pd.DataFrame(columns=['分子量','num_atom', 'LogP','QED', 'SA', 'Lipinski', 'Similarity','Diversity','Vina_score','Time','ligand'])
path_ori = r'/home/linux/Documents/DiffSBDD-Data/data_pre/processed_crossdock_noH_full_temp/test'
path_new = r'/home/linux/DiffFBDD/baseline/Pocket2Mol/results/'
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
df = pd.read_csv('/home/linux/DiffFBDD/baseline/Pocket2Mol/qvina_all/qvina2_scores.csv')

temp_full = pd.read_csv('/home/linux/PycharmProjects/DiffSBDD-main-before/DiffSBDD-main/result725_full.csv')
names = temp_full["ligand"]
names = list(set(names))

pdb_path = '/home/linux/Downloads/test_list_before.tsv'
path_time = '/home/linux/DiffFBDD/baseline/Pocket2Mol/outputs/'
sdf_name = [str(read_tsv(pdb_path)[int(i)]).split('/')[-1].split('.')[0].replace('_','-') for i in range(100)]
mean = []
for ori in file_ori:
    for new in sdf_name:
        resultsdd = {}
        if ori.split('/')[-1].split('.')[0].split('_')[1] == new:
            if sdf_name.index(new) in [92,3]:
                continue
            pocket_mols_sdf = '/home/linux/DiffFBDD/baseline/Pocket2Mol/results/'+str(sdf_name.index(new))+'.sdf'
            len1 = len(Chem.SDMolSupplier(pocket_mols_sdf))
            mols = Chem.SDMolSupplier(pocket_mols_sdf)
            for ss in range(len(df['ligand'])):
                if str(df['ligand'][ss]).split('/')[-1].split('.')[0] == str(sdf_name.index(new)):
                    vina = str(df['scores'][ss]).split('[')[1]
            for tt in range(len(or_l['ligand'])):
                if str(or_l['ligand'][tt]).split('/')[-1] == ori.split('/')[-1]:
                    vina_ori = or_l['scores'][tt]
            sum = []
            for i in range(len(Chem.SDMolSupplier(pocket_mols_sdf))):
                pocket_mols = Chem.SDMolSupplier(pocket_mols_sdf)[i]
                if pocket_mols:
                    Chem.SanitizeMol(pocket_mols)
                    # print(pocket_mols.GetNumAtoms())
                original_mol_sdf = ori
                original_mol = Chem.SDMolSupplier(original_mol_sdf)[0]
                part = 'sample_'+str(sdf_name.index(new))
                matching_dirs = [d for d in os.listdir(path_time) if d.startswith(part)]
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

                                         'Time' : read_txt(path_time+matching_dirs[0]+'/log.txt'),
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
                    # if iteration_result1['Vina_score'] < 0 and iteration_result1['num_atom'] > 1:
                    results_df = results_df.append(iteration_result1, ignore_index=True)
                        # results_df = results_df.append(iteration_result2, ignore_index=True)
                except:
                    i = i+1
                    continue
                sum.append(iteration_result1)
            mean.append(con_mean(sum))
        print(len(mean))

        for i in mean:
            print(i)

        csv_path = '/home/linux/DiffFBDD/baseline/Pocket2Mol/results_feature/mean.csv'
        with open(csv_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['分子量', 'num_atom', 'LogP', 'QED', 'SA', 'Lipinski',
                                                              'Similarity', 'Diversity', 'Vina_score', 'Time','ligand'])
            writer.writeheader()
            for row in mean:
                writer.writerow(row)
