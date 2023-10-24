import os
import sys
import glob
import fnmatch

def pdbs_to_pdbqts(pdb_dir, pdbqt_dir, dataset):
    print('----------------start--------------------------------')
    # print(pdb_dir, pdbqt_dir, dataset)
    # files = glob.glob(os.path.join(pdb_dir,'*.pdb'))
    # print(files)
    #
    # filr = os.path.join(pdb_dir,'*.pdb')
    # print(filr)
    #
    # fil = fnmatch.filter(os.listdir('/home/linux/Documents/processed_noH_full/test'), '*.pdb')
    # print(fil)
    for file in glob.glob(os.path.join(pdb_dir,'*.pdb')):
        print(file)
        name = os.path.splitext(os.path.basename(file))[0]
        outfile = os.path.join(pdbqt_dir, name + '.pdbqt')
        pdb_to_pdbqt(file, outfile, dataset)
        print('Wrote converted file to {}'.format(outfile))


def pdb_to_pdbqt(pdb_file, pdbqt_file, dataset):
    if dataset == 'crossdocked':
        os.system('prepare_receptor4.py -r {} -o {}'.format(pdb_file, pdbqt_file))
    elif dataset == 'bindingmoad':
        os.system('prepare_receptor4.py -r {} -o {} -A checkhydrogens -e'.format(pdb_file, pdbqt_file))
    else:
        raise NotImplementedError
    return pdbqt_file


if __name__ == '__main__':
    # print(sys.argv[1], sys.argv[2])
    pdbs_to_pdbqts(sys.argv[1], sys.argv[2], sys.argv[3])
