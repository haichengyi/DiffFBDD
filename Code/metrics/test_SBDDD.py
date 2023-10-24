import csv
from datetime import datetime

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolAlign, rdFMCS, QED, Crippen, Lipinski

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

def div(mols):
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

def wirte_mean(results_df,csv_path):
    result = []
    for i in results_df:
        if i == 'ligand':
            continue
        else:
            result.append(float(results_df[i].sum(axis=0)) / len(results_df['ligand']))
    mean = {'分子量': result[0],'num_atom':result[1],'LogP': result[2],
            'QED': result[3],'SA': result[4],'Lipinski': result[5],
            'Similarity': result[6],'Diversity': result[7],
            'Vina_score': result[8],'ligand': 'ALL'}
    print(mean)
    results_df = results_df.append(mean, ignore_index=True)
    results_df.to_csv(csv_path, index=False)

def cacular_file_lenth(paths):
    files_end = []
    for root, dirs, files in os.walk(path_ori):
        for file in files:
            path = os.path.join(root, file)
            if 'sdf' in file:
                files_end.append(path)
    return files_end

def read_txt(path):
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
def cacular_features(pocket_mol,mol,original_mol,path_time,part):
    matching_dirs = [d for d in os.listdir(path_time) if d.startswith(part)]
    feature = {'分子量': Descriptors.MolWt(pocket_mols),
     'num_atom': pocket_mols.GetNumAtoms(),
     'LogP': Descriptors.MolLogP(pocket_mols),
     'QED': QED.qed(pocket_mols),
     'SA': round((10 - calculateScore(pocket_mols)) / 9, 2),
     'Lipinski': rules(original_mol),
     'Similarity': sim(pocket_mols, original_mol),
     'Diversity': div(mols),
     'Vina_score': float(vina.split(',')[i].split(']')[0]),
     'Time':read_txt(path_time+matching_dirs[0]+'/log.txt'),
     'ligand': new.split('/')[-1]
     }
    return feature

def cacular_Top_1(max,item2):
    temp = {}
    print(max,item2)
    for key,value in max.items():
        for k,v in item2.items():
            if key == k:
                # print(k,max[key],item2[k])
                if key == 'Similarity' or 'Vina_score':
                    print(temp)
                    if(float(v)<float(value)):
                        temp[key] = item2[k]
                    else:
                        temp[key] = max[k]
                if key == 'QED' or 'SA' or 'Lipinski' or 'Diversity' or '分子量' or 'num_atom' or 'LogP':
                    if(float(v)>float(value)):
                        temp[key] = item2[k]
                    else:
                        temp[key] = max[k]
            else:
                continue
    return temp
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
def find_top_five_dicts(dict_list, key_to_compare):
    top_five_dicts = []
    for d in dict_list:
        if len(top_five_dicts) < 5:
            top_five_dicts.append(d)
        else:
            min_index = min(range(5), key=lambda i: top_five_dicts[i][key_to_compare])
            if d[key_to_compare] > top_five_dicts[min_index][key_to_compare]:
                top_five_dicts[min_index] = d
    return top_five_dicts


if __name__ == "__main__":

    results_df = pd.DataFrame(
        columns=['分子量', 'num_atom', 'LogP', 'QED', 'SA', 'Lipinski', 'Similarity', 'Diversity', 'Vina_score','Time', 'ligand'])

    path_ori = r'/home/linux/Documents/DiffSBDD-Data/data_pre/processed_crossdock_noH_full_temp/test'
    path_new = r'/home/linux/DiffFBDD/baseline/3DSBDD/results/'
    path_time = r'/home/linux/DiffFBDD/baseline/3DSBDD/outputs/'
    print(len(cacular_file_lenth(path_ori)),len(cacular_file_lenth(path_new)))

    # qvina
    or_l = pd.read_csv('/home/linux/R516/ori_vina/qvina2_scores.csv')  # crossDock
    df = pd.read_csv('/home/linux/DiffFBDD/baseline/3DSBDD/qvina_all/qvina2_scores.csv')

    pdb_path = '/home/linux/Downloads/test_list_before.tsv'
    sdf_name = [str(read_tsv(pdb_path)[int(i)]).split('/')[-1].split('.')[0].replace('_', '-') for i in range(100)]

    mean = []
    for ori in cacular_file_lenth(path_ori):
        for new in sdf_name:
            if ori.split('/')[-1].split('.')[0].split('_')[1] == new:
                pocket_mols_sdf = '/home/linux/DiffFBDD/baseline/3DSBDD/results/' + str(sdf_name.index(new)) + '.sdf'
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
                    try:
                        iteration_result1 = cacular_features(pocket_mols,mols,original_mol,path_time,'sample-'+str(sdf_name.index(new)))
                        # iteration_result2 = cacular_features(original_mol,original_mol,original_mol)
                        results_df = results_df.append(iteration_result1, ignore_index=True)
                    except:
                        i = i + 1
                        continue
                    sum.append(iteration_result1)
                mean.append(con_mean(sum))
    print(len(mean))

    for i in mean:
        print(i)

    csv_path = '/home/linux/DiffFBDD/baseline/3DSBDD/results_feature/mean.csv'
    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['分子量', 'num_atom', 'LogP', 'QED', 'SA', 'Lipinski', 'Similarity', 'Diversity', 'Vina_score', 'Time','ligand'])
        writer.writeheader()
        for row in mean:
            writer.writerow(row)
    # i.to_csv(csv_path,index=False)
    # results_df.to_csv(csv_path, index=False)
    # wirte_mean(results_df,csv_path)
