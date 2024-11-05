### Equivariant 3D-Conditional Diffusion Model for <i>De Novo</i> Drug Design

De novo drug design speeds up drug discovery, mitigating its time and cost burdens with advanced computational methods. Previous work either insufficiently utilized the 3D geometric structure of the target proteins, or generated ligands in an order that was inconsistent with real physics. Here we propose an equivariant 3D-conditional diffusion model, named DiffFBDD, for generating new pharmaceutical compounds based on 3D geometric information of specific target protein pockets. DiffFBDD overcomes the underutilization of geometric information by integrating full atomic information of pockets to backbone atoms using an equivariant graph neural network. Moreover, we develop a diffusion approach to generate drugs by generating ligand fragments for specific protein pockets, which requires fewer computational resources and less generation time (65.98% ∼ 96.10% lower). DiffFBDD offers better performance than state-of-the-art models in generating ligands with strong binding affinity to specific protein pockets, while maintaining high validity, uniqueness, and novelty, with clear potential for exploring the drug-like chemical space.

### Usage
#### Create a conda environment

```
conda env create -f environment.yml
```

#### QuickVina2

For docking, install QuickVina 2:

```
wget https://github.com/QVina/qvina/raw/master/bin/qvina2.1
chmod +x qvina2.1 
```

MGLTools for preparing the receptor for docking (pdb -> pdbqt) 

```
conda create -n mgltools -c bioconda mgltools
```

#### Training

Starting a new training run:

```
python -u train.py --config <config>.yml
```

Resuming a previous run:

```
python -u train.py --config <config>.yml --resume <checkpoint>.ckpt
```

#### Test

```
python test.py <checkpoint>.ckpt --test_dir <output_dir> --outdir <output_dir>
```

#### Metrics

Under the Metrics folder, verify SA, QED, Div, Time, LogP, Lipinski

#### QuickVina2

We follow the DiffSBDD method for verification. The verification method is as follows

First, convert all protein PDB files to PDBQT files using MGLTools

```
conda activate mgltools
cd analysis
python docking_py27.py <test_dir> <output_dir>
cd ..
conda deactivate
```

Then, compute QuickVina scores:

```
conda activate diff-fbdd
python analysis/docking.py --pdbqt_dir <docking_py27_outdir> --sdf_dir <test_outdir> --out_dir <qvina_outdir> --write_csv --write_dict
```

### Citation
Please cite this work as below：
```
@ARTICLE{DiffFBDD,
  author={Zheng, Jia and Yi, Hai-Cheng and You, Zhu-Hong},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Equivariant 3D-Conditional Diffusion Model for De Novo Drug Design}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  keywords={Atoms;Proteins;Drugs;Three-dimensional displays;Chemicals;Bioinformatics;Diffusion models;Graph neural networks;Point cloud compression;Drug discovery;Artificial intelligence;drug discovery;equivariant diffusion model;fragment-based drug design;molecular generation},
  doi={10.1109/JBHI.2024.3491318}}
```
