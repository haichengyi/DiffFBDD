import os
import re
import torch
from pathlib import Path
import argparse

import pandas as pd
from rdkit import Chem
from tqdm import tqdm


def calculate_smina_score(pdb_file, sdf_file):
    # add '-o <name>_smina.sdf' if you want to see the output
    out = os.popen(f'smina.static -l {sdf_file} -r {pdb_file} '
                   f'--score_only').read()
    matches = re.findall(
        r"Affinity:[ ]+([+-]?[0-9]*[.]?[0-9]+)[ ]+\(kcal/mol\)", out)
    return [float(x) for x in matches]


def sdf_to_pdbqt(sdf_file, pdbqt_outfile, mol_id):
    os.popen(f'obabel {sdf_file} -O {pdbqt_outfile} '
             f'-f {mol_id + 1} -l {mol_id + 1}').read()
    return pdbqt_outfile


def calculate_qvina2_score(receptor_file, sdf_file, out_dir, size=20,
                           exhaustiveness=16, return_rdmol=False):

    receptor_file = Path(receptor_file)
    sdf_file = Path(sdf_file)

    if receptor_file.suffix == '.pdb':
        # prepare receptor, requires Python 2.7
        receptor_pdbqt_file = Path(out_dir, receptor_file.stem + '.pdbqt')
        os.popen(f'prepare_receptor4.py -r {receptor_file} -O {receptor_pdbqt_file}')
    else:
        receptor_pdbqt_file = receptor_file

    scores = []
    rdmols = []  # for if return rdmols
    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)

    for i, mol in enumerate(suppl):  # sdf file may contain several ligands
        ligand_name = f'{sdf_file.stem}_{i}'
        # prepare ligand
        ligand_pdbqt_file = Path(out_dir, ligand_name + '.pdbqt')
        out_sdf_file = Path(out_dir, ligand_name + '_out.sdf')

        if out_sdf_file.exists():
            with open(out_sdf_file, 'r') as f:
                scores.append(
                    min([float(x.split()[2]) for x in f.readlines()
                         if x.startswith(' VINA RESULT:')])
                )

        else:
            sdf_to_pdbqt(sdf_file, ligand_pdbqt_file, i)

            # center box at ligand's center of mass
            cx, cy, cz = mol.GetConformer().GetPositions().mean(0)
            # print('shizhema')
            # run QuickVina 2
            out = os.popen(
                f'qvina2.1 --receptor {receptor_pdbqt_file} '
                f'--ligand {ligand_pdbqt_file} '
                f'--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} '
                f'--size_x {size} --size_y {size} --size_z {size} '
                f'--exhaustiveness {exhaustiveness}'
            ).read()
            # print('shizhema')
            out_split = out.splitlines()
            best_idx = out_split.index('-----+------------+----------+----------') + 1
            best_line = out_split[best_idx].split()
            assert best_line[0] == '1'
            scores.append(float(best_line[1]))
            print(scores)

            out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
            if out_pdbqt_file.exists():
                os.popen(f'obabel {out_pdbqt_file} -O {out_sdf_file}').read()

        if return_rdmol:
            try:
                rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
                rdmols.append(rdmol)
            except:
                rdmol = None

    if return_rdmol:
        return scores, rdmols
    else:
        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser('QuickVina evaluation')
    parser.add_argument('--pdbqt_dir', type=Path,
                        help='Receptor files in pdbqt format')
    parser.add_argument('--sdf_dir', type=Path, default=None,
                        help='Ligand files in sdf format')
    parser.add_argument('--sdf_files', type=Path, nargs='+', default=None)
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--write_csv', action='store_true')
    parser.add_argument('--write_dict', action='store_true')
    parser.add_argument('--dataset', type=str, default='moad')
    args = parser.parse_args()

    assert (args.sdf_dir is not None) ^ (args.sdf_files is not None)

    results = {'receptor': [], 'ligand': [], 'scores': []}
    results_dict = {}
    sdf_files = list(args.sdf_dir.glob('[!.]*.sdf')) \
        if args.sdf_dir is not None else args.sdf_files
    pbar = tqdm(sdf_files)
    for sdf_file in pbar:
        pbar.set_description(f'Processing {sdf_file.name}')

        if args.dataset == 'moad':
            """
            Ligand file names should be of the following form:
            <receptor-name>_<pocket-id>_<some-suffix>.sdf
            where <receptor-name> and <pocket-id> cannot contain any 
            underscores, e.g.: 1abc-bio1_pocket0_gen.sdf
            """
            ligand_name = sdf_file.stem

            receptor_name, pocket_id, *suffix = ligand_name.split('_')
            suffix = '_'.join(suffix)
            temp = ligand_name.split('_')
            receptor_name = temp[0] + '_' + temp[1]
            print(receptor_name)
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')
        elif args.dataset == 'crossdocked':
            ligand_name = sdf_file.stem
            receptor_name = ligand_name[:-4]
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')
        elif args.dataset == 'new':
            ligand_name = sdf_file.stem
            receptor_name = '1p9s'
            receptor_file = Path(args.pdbqt_dir, receptor_name + '.pdbqt')
        try:
            scores, rdmols = calculate_qvina2_score(
                receptor_file, sdf_file, args.out_dir, return_rdmol=True)
        except (ValueError, AttributeError) as e:
            print(e)
            continue
        results['receptor'].append(str(receptor_file))
        results['ligand'].append(str(sdf_file))
        results['scores'].append(scores)
        # print(results)

        if args.write_dict:
            results_dict[receptor_name] = [scores, rdmols]
    # print('write csv')
    # print(results)
    if args.write_csv:
        df = pd.DataFrame.from_dict(results)
        df.to_csv(Path(args.out_dir, 'qvina2_scores.csv'))

    if args.write_dict:
        torch.save(results_dict, Path(args.out_dir, 'qvina2_scores.pt'))
