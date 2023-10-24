import math
from argparse import Namespace
from typing import Optional
from time import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
from torch_scatter import scatter_add, scatter_mean
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one

from constants import dataset_params, FLOAT_TYPE, INT_TYPE
from equivariant_diffusion.dynamics import EGNNDynamics
from equivariant_diffusion.conditional_model import ConditionalDDPM
from dataset import ProcessedLigandPocketDataset
import utils
from analysis.visualization import save_xyz_file, visualize, visualize_chain
from analysis.metrics import check_stability, BasicMolecularMetrics, \
    CategoricalDistribution
from analysis.molecule_builder import build_molecule, process_molecule

from torch.distributions import MultivariateNormal,Categorical

class LigandPocketFBDD(pl.LightningModule):
    # 初始化函数
    def __init__(
            self,outdir,dataset,datadir,batch_size,lr,egnn_params: Namespace,diffusion_params,
            num_workers,augment_noise,augment_rotation,clip_grad,eval_epochs,eval_params,
            visualize_sample_epoch,visualize_chain_epoch,mode,node_histogram,pocket_representation='CA',
    ):
        super(LigandPocketFBDD, self).__init__()
        self.save_hyperparameters()
        ddpm_models ={ 'pocket_conditioning':ConditionalDDPM}
        self.mode = mode
        assert pocket_representation in {'CA', 'full-atom'}
        self.pocket_representation = pocket_representation
        self.dataset_name = dataset
        self.datadir = datadir
        self.outdir = outdir
        self.batch_size = batch_size
        self.eval_batch_size = eval_params.eval_batch_size \
            if 'eval_batch_size' in eval_params else batch_size
        self.lr = lr
        self.loss_type = diffusion_params.diffusion_loss_type
        self.eval_epochs = eval_epochs
        self.visualize_sample_epoch = visualize_sample_epoch
        self.visualize_chain_epoch = visualize_chain_epoch
        self.eval_params = eval_params
        self.num_workers = num_workers
        self.augment_noise = augment_noise
        self.augment_rotation = augment_rotation
        self.dataset_info = dataset_params[dataset]
        self.T = diffusion_params.diffusion_steps
        self.clip_grad = clip_grad
        if clip_grad:
            self.gradnorm_queue = utils.Queue()
            # Add large value that will be flushed.
            self.gradnorm_queue.add(3000)

        smiles_list = None if eval_params.smiles_file is None \
            else np.load(eval_params.smiles_file)
        self.ligand_metrics = BasicMolecularMetrics(self.dataset_info,
                                                    smiles_list)
        self.ligand_type_distribution = CategoricalDistribution(
            self.dataset_info['atom_hist'], self.dataset_info['atom_encoder'])
        if self.pocket_representation == 'CA':
            self.pocket_type_distribution = CategoricalDistribution(
                self.dataset_info['aa_hist'], self.dataset_info['aa_encoder'])
        else:
            # TODO: full-atom case
            self.pocket_type_distribution = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.lig_type_encoder = self.dataset_info['atom_encoder']
        self.lig_type_decoder = self.dataset_info['atom_decoder']
        self.pocket_type_encoder = self.dataset_info['aa_encoder'] \
            if self.pocket_representation == 'CA' \
            else self.dataset_info['atom_encoder']
        self.pocket_type_decoder = self.dataset_info['aa_decoder'] \
            if self.pocket_representation == 'CA' \
            else self.dataset_info['atom_decoder']

        self.atom_nf = len(self.lig_type_decoder)
        self.aa_nf = len(self.pocket_type_decoder)
        self.x_dims = 3

        net_dynamics = EGNNDynamics(
            atom_nf=self.atom_nf,
            residue_nf=self.aa_nf,
            n_dims=self.x_dims,
            joint_nf=egnn_params.joint_nf,
            device=egnn_params.device if torch.cuda.is_available() else 'cpu',
            hidden_nf=egnn_params.hidden_nf,
            act_fn=torch.nn.SiLU(),
            n_layers=egnn_params.n_layers,
            attention=egnn_params.attention,
            tanh=egnn_params.tanh,
            norm_constant=egnn_params.norm_constant,
            inv_sublayers=egnn_params.inv_sublayers,
            sin_embedding=egnn_params.sin_embedding,
            normalization_factor=egnn_params.normalization_factor,
            aggregation_method=egnn_params.aggregation_method,
            edge_cutoff=egnn_params.__dict__.get('edge_cutoff'),
            update_pocket_coords=(self.mode == 'joint')
        )

        self.ddpm = ddpm_models[self.mode](
            dynamics=net_dynamics,
            atom_nf=self.atom_nf,
            residue_nf=self.aa_nf,
            n_dims=self.x_dims,
            timesteps=diffusion_params.diffusion_steps,
            noise_schedule=diffusion_params.diffusion_noise_schedule,
            noise_precision=diffusion_params.diffusion_noise_precision,
            loss_type=diffusion_params.diffusion_loss_type,
            norm_values=diffusion_params.normalize_factors,
            size_histogram=node_histogram,
        )

    # 优化器：loss值梯度下降优化器--输入一个需要迭代的参数，学习率，可以优化权重
    def configure_optimizers(self):
        return torch.optim.AdamW(self.ddpm.parameters(), lr=self.lr,
                                 amsgrad=True, weight_decay=1e-12)
    # 启动函数:控制训练与测试
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'train.npz'))
            self.val_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'val.npz'))
        elif stage == 'test':
            self.test_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'test.npz'))
        else:
            raise NotImplementedError
    # 训练集下载
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.train_dataset.collate_fn)
    #验证集下载
    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.val_dataset.collate_fn)
    # 测试集下载
    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn)
    # 各个数据集的数据元素获取
    def get_ligand_and_pocket(self, data):
        ligand = {
            'x': data['lig_coords'].to(self.device, FLOAT_TYPE),
            'one_hot': data['lig_one_hot'].to(self.device, FLOAT_TYPE),
            'size': data['num_lig_atoms'].to(self.device, INT_TYPE),
            'mask': data['lig_mask'].to(self.device, INT_TYPE)
        }

        pocket = {
            'x': data['pocket_c_alpha_ca'].to(self.device, FLOAT_TYPE),
            'one_hot': data['pocket_one_hot_ca'].to(self.device, FLOAT_TYPE),
            'size': data['num_pocket_nodes'].to(self.device, INT_TYPE),
            'mask': data['pocket_mask_ca'].to(self.device, INT_TYPE),
            'x_full':data['pocket_c_alpha_full'].to(self.device,FLOAT_TYPE),
            'one_hot_full':data['pocket_c_one_hot_full'].to(self.device,FLOAT_TYPE),
            'mask_full':data['pocket_mask_full'].to(self.device,INT_TYPE),
            'size_full':data['num_pocket_nodes_full'].to(self.device,INT_TYPE)
        }
        return ligand, pocket
    # LLPM的训练函数主要调用：
    # get_ligand_and_pocket--获取数据
    # self.ddpm--训练数据，返回部分loss结果
    # 计算Lsimple--公式3  Loss = L0+Lt+Ls(s<t)累加
    # 主要的存在的loss计算方法有
    # （1）l2:L2范数损失函数，也被称为最小平方误差（LSE）。总的来说，它是把目标值（Yi)与估计值（f(xi))的差值的平方和（S)最小化：
    # （2）vlb:变分下界损失函数--当反向传播过程中的方差是可学习的时候会用到
    # （3）weight_Lj的新的损失函数参数--liner(此处注意！）
    # 最后返回整个损失的结果Loss+info
    def apart_data(self,combine_tensor,mask):

        num_components = 2
        input_dim = combine_tensor.shape[1]
        weights = (torch.ones(num_components) / num_components).to('cuda')
        means = (torch.randn(num_components,input_dim)).to('cuda')
        covs = (torch.eye(input_dim).unsqueeze(0).repeat(num_components,1,1)).to('cuda')

        gaussian_mixture = Categorical(weights)
        component_distributions = MultivariateNormal(means,covs)
        num_steps = 100
        optimizer = torch.optim.Adam([means,covs],lr = 0.01)

        cluster_indices1 = []
        cluster_indices2 = []

        unique_masks = torch.unique(mask)  # 获取唯一的mask值

        for unique_mask in unique_masks:

            mask_indices = torch.where(mask == unique_mask)[0]
            combine_tensor_subset = combine_tensor[mask_indices]

            for step in range(num_steps):
                # E步：计算每个样本属于每个聚类簇的概率
                log_probs = gaussian_mixture.logits + component_distributions.log_prob(combine_tensor_subset.unsqueeze(1))
                responsibilities = torch.softmax(log_probs, dim=1)

                # M步：更新模型参数
                optimizer.zero_grad()
                loss = -responsibilities.sum(dim=0).mean()  # 最大化似然函数的负数作为损失函数
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()

            # 根据聚类结果将数据分成两类
            _, cluster_indices = responsibilities.max(dim=1)
            cluster_indices_1 = (torch.where(cluster_indices == 0)[0]).cuda()
            cluster_indices_2 = (torch.where(cluster_indices == 1)[0]).cuda()

            cluster_indices1.append(cluster_indices_1)
            cluster_indices2.append(cluster_indices_2)

        for i in range(len(cluster_indices1)):
            if len(cluster_indices1[i]) == 0:
                return self.apart_data(combine_tensor, mask)
        for i in range(len(cluster_indices2)):
            if len(cluster_indices2[i]) == 0:
                return self.apart_data(combine_tensor, mask)
        # if len(cluster_indices1[0]) == 0 or len(cluster_indices2[0]) == 0:
        #     return self.apart_data(combine_tensor,mask)

        return cluster_indices1,cluster_indices2

    def split_ligand_by_index(self,ligand, result1,result2,len12,len22):
        ligand1 = {}
        ligand2 = {}
        sum1 = 0
        sum2 = 0
        # print(result1,result2)
        for i in range(len(len12)):
            if len12[i] == 0:
                # print('Ligand has zero')
                # print(len(result1), len(result2), len12[i], len22[i],i)
                # print('after!')
                sss = result2[sum2].unsqueeze(dim = 0)
                result1 = torch.cat((result1[:sum1],sss,result1[sum1:]),dim = 0)
                result2 = torch.cat((result2[:sum2],result2[sum2+1:]))
                # print(len(result1), len(result2),len12[i],len22[i])
                len12[i] = 1
                len22[i] = len22[i] - 1
            if len22[i] == 0:
                # print('Ligand has zero')
                # print(len(result1), len(result2), len12[i], len22[i], i)
                # print('after!')
                sss = result1[sum1].unsqueeze(0)
                result2 = torch.cat((result2[:sum2], sss, result2[sum2:]),dim = 0)
                result1 = torch.cat((result1[:sum1], result1[sum1 + 1:]))
                len22[i] = 1
                len12[i] = len12[i] - 1
                # print(len(result1), len(result2), len12[i], len22[i])
            sum1 += len12[i]
            sum2 += len22[i]
        for s in {'x','one_hot','mask'}:
            ligand1[s] = ligand[s][result1]
            ligand2[s] = ligand[s][result2]
        ligand1['size'] = torch.tensor(len12).long().cuda()
        ligand2['size'] = torch.tensor(len22).long().cuda()
        return ligand1,ligand2
    def split_pocket_by_index(self,pocket, result1,result2,len11,len21):
        pocket1 = {}
        pocket2 = {}
        for s in {'x','one_hot','mask'}:
            pocket1[s] = pocket[s][result1]
            pocket2[s] = pocket[s][result2]
        pocket1['size'] = torch.tensor(len11).long().cuda()
        pocket2['size'] = torch.tensor(len21).long().cuda()
        # print(pocket)
        for key in {'x_full','one_hot_full','mask_full','size_full'}:
            pocket1[key] = pocket[key]
            pocket2[key] = pocket[key]
        return pocket1,pocket2
    def result_apart(self,cluster_indices,num_atoms,num_pockets):
        j = 0
        apart1 = []
        apart2 = []
        for i in cluster_indices:
            mask = i >= num_atoms[j]
            apart1.append(torch.masked_select(i, mask))
            apart2.append(torch.masked_select(i, ~mask))
            j = j+1
        apart_pocket = []
        sum = 0
        j = 0
        len1 = []
        len2 = []
        for i in apart1:
            apart_pocket.append(i-num_atoms[j] + sum)
            len1.append(len(i))
            sum += num_pockets[j]
            j = j + 1
        apart_ligand = []
        sum = 0
        j = 0
        for i in apart2:
            apart_ligand.append(i+sum)
            len2.append(len(i))
            sum += num_atoms[j]
            j = j+1
        return torch.cat(apart_pocket,dim = 0),torch.cat(apart_ligand,dim = 0),len1,len2
    def merge_ligand(self,ligand1,ligand2):
        ligand = {}
        for item in {'x','one_hot','mask','size'}:
            if item == 'size':
                ligand[item] = ligand1[item]+ligand2[item]
                continue
            ligand[item] = torch.cat((ligand1[item],ligand2[item]),dim=0)
        return ligand
    def merge_pocket(self,pocket1,pocket2):
        pocket = {}
        for item in {'x','one_hot','mask','size'}:
            if item == 'size':
                pocket[item] = pocket1[item]+pocket2[item]
                continue
            pocket[item] = torch.cat((pocket1[item],pocket2[item]),dim=0)
        for key in {'x_full','one_hot_full','mask_full','size_full'}:
            pocket[key] = pocket1[key]
        return pocket
    def forward(self, data):
        ligand, pocket = self.get_ligand_and_pocket(data)
        # use GMM to apart
        x_atoms = ligand['x'][:, :3].clone()
        x_residues = pocket['x'][:, :3].clone()
        x = torch.cat((x_atoms, x_residues), dim=0)
        mask = torch.cat([ligand['mask'],pocket['mask']])
        cluster_indices1, cluster_indices2 = self.apart_data(x,mask)

        result_pocket_1,result_ligand_1,len11,len12 = self.result_apart(cluster_indices1,ligand['size'],pocket['size'])
        result_pocket_2, result_ligand_2,len21,len22 = self.result_apart(cluster_indices2, ligand['size'],pocket['size'])

        ligand1,ligand2 = self.split_ligand_by_index(ligand,result_ligand_1,result_ligand_2,len12,len22)
        pocket1, pocket2 = self.split_pocket_by_index(pocket, result_pocket_1, result_pocket_2,len11,len21)

        delta_log_px, error_t_lig, error_t_pocket, SNR_weight, \
            loss_0_x_ligand, loss_0_x_pocket, loss_0_h, neg_log_const_0, \
            kl_prior, log_pN, t_int, xh_lig_hat, loss_0_for_new,info = \
            self.ddpm(ligand1,pocket1,ligand2, pocket2, return_info=True)

        ligand = self.merge_ligand(ligand1,ligand2)
        pocket = self.merge_pocket(pocket1,pocket2)
        if self.loss_type == 'l2' and self.training:
            # normalize loss_t
            denom_lig = (self.x_dims + self.ddpm.atom_nf) * ligand['size']
            error_t_lig = error_t_lig / denom_lig
            denom_pocket = (self.x_dims + self.ddpm.residue_nf) * pocket['size']
            error_t_pocket = error_t_pocket / denom_pocket
            # L_simple
            loss_t = 0.5 * (error_t_lig + error_t_pocket)

            # normalize loss_0
            loss_0_x_ligand = loss_0_x_ligand / (self.x_dims * ligand['size'])
            loss_0_x_pocket = loss_0_x_pocket / (self.x_dims * pocket['size'])
            loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h

        # VLB objective or evaluation step
        else:
            # Note: SNR_weight should be negative
            loss_t = -self.T * 0.5 * SNR_weight * (error_t_lig + error_t_pocket)
            loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h
            loss_0 = loss_0 + neg_log_const_0
        # 这里是用来计算loss的值（公式（3））
        # nll = loss_t + loss_0 + kl_prior + loss_0_for_new
        nll = loss_t + loss_0 + kl_prior
        # Correct for normalization on x.
        if not (self.loss_type == 'l2' and self.training):
            nll = nll - delta_log_px

            # Transform conditional nll into joint nll
            # Note:
            # loss = -log p(x,h|N) and log p(x,h,N) = log p(x,h|N) + log p(N)
            # Therefore, log p(x,h|N) = -loss + log p(N)
            # => loss_new = -log p(x,h,N) = loss - log p(N)
            nll = nll - log_pN

        info['error_t_lig'] = error_t_lig.mean(0)
        info['error_t_pocket'] = error_t_pocket.mean(0)
        info['SNR_weight'] = SNR_weight.mean(0)
        info['loss_0'] = loss_0.mean(0)
        info['kl_prior'] = kl_prior.mean(0)
        info['delta_log_px'] = delta_log_px.mean(0)
        info['neg_log_const_0'] = neg_log_const_0.mean(0)
        info['log_pN'] = log_pN.mean(0)
        info['loss_0_for_new'] = loss_0_for_new.mean(0)

        # print(nll,info)
        # return 0
        return nll, info
    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)
    # 数据训练：将loss值以及一些详细值进行返回
    def training_step(self, data, *args):
        # 这的目的是什么不知道
        if self.augment_noise > 0:
            raise NotImplementedError
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian(x.size(), x.device)
            x = x + eps * args.augment_noise

        if self.augment_rotation:
            raise NotImplementedError
            x = utils.random_rotation(x).detach()

        nll, info = self.forward(data)
        loss = nll.mean(0)

        info['loss'] = loss
        self.log_metrics(info, 'train', batch_size=len(data['num_lig_atoms']))

        return info
    # val和test共享的用来评估模型的损失
    def _shared_eval(self, data, prefix, *args):
        nll, info = self.forward(data)
        loss = nll.mean(0)

        info['loss'] = loss

        # some additional info
        gamma_0 = self.ddpm.gamma(torch.zeros(1, device=self.device))
        gamma_1 = self.ddpm.gamma(torch.ones(1, device=self.device))
        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1
        info['log_SNR_max'] = log_SNR_max
        info['log_SNR_min'] = log_SNR_min

        self.log_metrics(info, prefix, batch_size=len(data['num_lig_atoms']),
                         sync_dist=True)

        return info
    # 验证集步骤
    def validation_step(self, data, *args):
        self._shared_eval(data, 'val', *args)
    # 测试集步骤--两个是相同的，此处不需要进行更改
    def test_step(self, data, *args):
        self._shared_eval(data, 'test', *args)
    # 每n（自己设置）次进行一次验证并采样
    def validation_epoch_end(self, validation_step_outputs):

        # Perform validation on single GPU
        # TODO: sample on multiple devices if available
        if not self.trainer.is_global_zero:
            return

        suffix = '' if self.mode == 'joint' else '_given_pocket'

        if (self.current_epoch + 1) % self.eval_epochs == 0:
            tic = time()
            # 等到验证集的epoch到了以后，开始进行采样
            sampling_results = getattr(self, 'sample_and_analyze' + suffix)(
                self.eval_params.n_eval_samples, self.val_dataset,
                batch_size=self.eval_batch_size)
            # 对生成的原子进行评估
            self.log_metrics(sampling_results, 'val')

            print(f'Evaluation took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_sample_epoch == 0:
            tic = time()
            getattr(self, 'sample_and_save' + suffix)(
                self.eval_params.n_visualize_samples)
            print(f'Sample visualization took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_chain_epoch == 0:
            tic = time()
            getattr(self, 'sample_chain_and_save' + suffix)(
                self.eval_params.keep_frames)
            print(f'Chain visualization took {time() - tic:.2f} seconds')
    # 采样并分析
    @torch.no_grad()
    def sample_and_analyze(self, n_samples, dataset=None, batch_size=None):
        print(f'Analyzing molecule stability at epoch {self.current_epoch}...')

        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, n_samples)

        # each item in molecules is a tuple (position, atom_type_encoded)
        molecules = []
        atom_types = []
        aa_types = []
        for i in range(math.ceil(n_samples / batch_size)):
            n_samples_batch = min(batch_size, n_samples - len(molecules))

            num_nodes_lig, num_nodes_pocket = \
                self.ddpm.size_distribution.sample(n_samples_batch)

            xh_lig, xh_pocket, lig_mask, _ = self.ddpm.sample(
                n_samples_batch, num_nodes_lig, num_nodes_pocket,
                device=self.device)

            x = xh_lig[:, :self.x_dims].detach().cpu()
            atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()

            molecules.extend(list(
                zip(utils.batch_to_list(x, lig_mask),
                    utils.batch_to_list(atom_type, lig_mask))
            ))

            atom_types.extend(atom_type.tolist())
            aa_types.extend(
                xh_pocket[:, self.x_dims:].argmax(1).detach().cpu().tolist())

        return self.analyze_sample(molecules, atom_types, aa_types)

    def analyze_sample(self, molecules, atom_types, aa_types):
        # Distribution of node types
        kl_div_atom = self.ligand_type_distribution.kl_divergence(atom_types) \
            if self.ligand_type_distribution is not None else -1
        kl_div_aa = self.pocket_type_distribution.kl_divergence(aa_types) \
            if self.pocket_type_distribution is not None else -1

        # Stability
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        for pos, atom_type in molecules:
            validity_results = check_stability(pos, atom_type,
                                               self.dataset_info)
            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])

        fraction_mol_stable = molecule_stable / float(len(molecules))
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)

        # Other basic metrics
        validity, connectivity, uniqueness, novelty = \
            self.ligand_metrics.evaluate(molecules)[0]

        return {
            'kl_div_atom_types': kl_div_atom,
            'kl_div_residue_types': kl_div_aa,
            'mol_stable': fraction_mol_stable,
            'atm_stable': fraction_atm_stable,
            'Validity': validity,
            'Connectivity': connectivity,
            'Uniqueness': uniqueness,
            'Novelty': novelty
        }

    @torch.no_grad()
    def sample_and_analyze_given_pocket(self, n_samples, dataset=None,
                                        batch_size=None):
        print(f'Analyzing molecule stability given pockets at epoch '
              f'{self.current_epoch}...')

        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, n_samples)

        # each item in molecules is a tuple (position, atom_type_encoded)
        molecules = []
        atom_types = []
        aa_types = []
        for i in range(math.ceil(n_samples / batch_size)):
            n_samples_batch = min(batch_size, n_samples - len(molecules))

            # Create a batch
            batch = dataset.collate_fn(
                [dataset[(i * batch_size + j) % len(dataset)]
                 for j in range(n_samples_batch)]
            )
            # 获取数据
            ligand, pocket = self.get_ligand_and_pocket(batch)
            # 根据pocket的大小在原始的大小分布上得出应该生成的配体的大小
            num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])
            # 从口袋中采样（均匀采样），并得到最终采样的x0
            xh_lig, xh_pocket, lig_mask, _ = self.ddpm.sample_given_pocket(
                pocket, num_nodes_lig)

            x = xh_lig[:, :self.x_dims].detach().cpu()
            atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()

            molecules.extend(list(
                zip(utils.batch_to_list(x, lig_mask),
                    utils.batch_to_list(atom_type, lig_mask))
            ))

            atom_types.extend(atom_type.tolist())
            aa_types.extend(
                xh_pocket[:, self.x_dims:].argmax(1).detach().cpu().tolist())

        return self.analyze_sample(molecules, atom_types, aa_types)

    def sample_and_save(self, n_samples):
        num_nodes_lig, num_nodes_pocket = \
            self.ddpm.size_distribution.sample(n_samples)

        xh_lig, xh_pocket, lig_mask, pocket_mask = \
            self.ddpm.sample(n_samples, num_nodes_lig, num_nodes_pocket,
                             device=self.device)

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                xh_pocket[:, :self.x_dims], self.dataset_info)
        else:
            x_pocket, one_hot_pocket = \
                xh_pocket[:, :self.x_dims], xh_pocket[:, self.x_dims:]
        x = torch.cat((xh_lig[:, :self.x_dims], x_pocket), dim=0)
        one_hot = torch.cat((xh_lig[:, self.x_dims:], one_hot_pocket), dim=0)

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')
        save_xyz_file(str(outdir) + '/', one_hot, x, self.dataset_info,
                      name='molecule',
                      batch_mask=torch.cat((lig_mask, pocket_mask)))
        # visualize(str(outdir), dataset_info=self.dataset_info, wandb=wandb)
        visualize(str(outdir), dataset_info=self.dataset_info, wandb=None)

    def sample_and_save_given_pocket(self, n_samples):
        batch = self.val_dataset.collate_fn(
            [self.val_dataset[i] for i in torch.randint(len(self.val_dataset),
                                                        size=(n_samples,))]
        )
        ligand, pocket = self.get_ligand_and_pocket(batch)

        num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
            n1=None, n2=pocket['size'])

        xh_lig, xh_pocket, lig_mask, pocket_mask = \
            self.ddpm.sample_given_pocket(pocket, num_nodes_lig)

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                xh_pocket[:, :self.x_dims], self.dataset_info)
        else:
            x_pocket, one_hot_pocket = \
                xh_pocket[:, :self.x_dims], xh_pocket[:, self.x_dims:]
        x = torch.cat((xh_lig[:, :self.x_dims], x_pocket), dim=0)
        one_hot = torch.cat((xh_lig[:, self.x_dims:], one_hot_pocket), dim=0)

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')
        save_xyz_file(str(outdir) + '/', one_hot, x, self.dataset_info,
                      name='molecule',
                      batch_mask=torch.cat((lig_mask, pocket_mask)))
        # visualize(str(outdir), dataset_info=self.dataset_info, wandb=wandb)
        visualize(str(outdir), dataset_info=self.dataset_info, wandb=None)

    def sample_chain_and_save(self, keep_frames):
        n_samples = 1
        n_tries = 1

        num_nodes_lig, num_nodes_pocket = \
            self.ddpm.size_distribution.sample(n_samples)

        one_hot_lig, x_lig, one_hot_pocket, x_pocket = [None] * 4
        for i in range(n_tries):
            chain_lig, chain_pocket, _, _ = self.ddpm.sample(
                n_samples, num_nodes_lig, num_nodes_pocket,
                return_frames=keep_frames, device=self.device)

            chain_lig = utils.reverse_tensor(chain_lig)
            chain_pocket = utils.reverse_tensor(chain_pocket)

            # Repeat last frame to see final sample better.
            chain_lig = torch.cat([chain_lig, chain_lig[-1:].repeat(10, 1, 1)],
                                  dim=0)
            chain_pocket = torch.cat(
                [chain_pocket, chain_pocket[-1:].repeat(10, 1, 1)], dim=0)

            # Check stability of the generated ligand
            x_final = chain_lig[-1, :, :self.x_dims].cpu().detach().numpy()
            one_hot_final = chain_lig[-1, :, self.x_dims:]
            atom_type_final = torch.argmax(
                one_hot_final, dim=1).cpu().detach().numpy()

            mol_stable = check_stability(x_final, atom_type_final,
                                         self.dataset_info)[0]

            # Prepare entire chain.
            x_lig = chain_lig[:, :, :self.x_dims]
            one_hot_lig = chain_lig[:, :, self.x_dims:]
            one_hot_lig = F.one_hot(
                torch.argmax(one_hot_lig, dim=2),
                num_classes=len(self.lig_type_decoder))
            x_pocket = chain_pocket[:, :, :self.x_dims]
            one_hot_pocket = chain_pocket[:, :, self.x_dims:]
            one_hot_pocket = F.one_hot(
                torch.argmax(one_hot_pocket, dim=2),
                num_classes=len(self.pocket_type_decoder))

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                x_pocket, self.dataset_info)

        x = torch.cat((x_lig, x_pocket), dim=1)
        one_hot = torch.cat((one_hot_lig, one_hot_pocket), dim=1)

        # flatten (treat frame (chain dimension) as batch for visualization)
        x_flat = x.view(-1, x.size(-1))
        one_hot_flat = one_hot.view(-1, one_hot.size(-1))
        mask_flat = torch.arange(x.size(0)).repeat_interleave(x.size(1))

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}', 'chain')
        save_xyz_file(str(outdir), one_hot_flat, x_flat, self.dataset_info,
                      name='/chain', batch_mask=mask_flat)
        visualize_chain(str(outdir), self.dataset_info, wandb=wandb)

    def sample_chain_and_save_given_pocket(self, keep_frames):
        n_samples = 1
        n_tries = 1

        batch = self.val_dataset.collate_fn([
            self.val_dataset[torch.randint(len(self.val_dataset), size=(1,))]
        ])
        ligand, pocket = self.get_ligand_and_pocket(batch)

        num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
            n1=None, n2=pocket['size'])

        one_hot_lig, x_lig, one_hot_pocket, x_pocket = [None] * 4
        for i in range(n_tries):
            chain_lig, chain_pocket, _, _ = self.ddpm.sample_given_pocket(
                pocket, num_nodes_lig, return_frames=keep_frames)

            chain_lig = utils.reverse_tensor(chain_lig)
            chain_pocket = utils.reverse_tensor(chain_pocket)

            # Repeat last frame to see final sample better.
            chain_lig = torch.cat([chain_lig, chain_lig[-1:].repeat(10, 1, 1)],
                                  dim=0)
            chain_pocket = torch.cat(
                [chain_pocket, chain_pocket[-1:].repeat(10, 1, 1)], dim=0)

            # Check stability of the generated ligand
            x_final = chain_lig[-1, :, :self.x_dims].cpu().detach().numpy()
            one_hot_final = chain_lig[-1, :, self.x_dims:]
            atom_type_final = torch.argmax(
                one_hot_final, dim=1).cpu().detach().numpy()

            mol_stable = check_stability(x_final, atom_type_final,
                                         self.dataset_info)[0]

            # Prepare entire chain.
            x_lig = chain_lig[:, :, :self.x_dims]
            one_hot_lig = chain_lig[:, :, self.x_dims:]
            one_hot_lig = F.one_hot(
                torch.argmax(one_hot_lig, dim=2),
                num_classes=len(self.lig_type_decoder))
            x_pocket = chain_pocket[:, :, :3]
            one_hot_pocket = chain_pocket[:, :, 3:]
            one_hot_pocket = F.one_hot(
                torch.argmax(one_hot_pocket, dim=2),
                num_classes=len(self.pocket_type_decoder))

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

        if self.pocket_representation == 'CA':
            # convert residues into atom representation for visualization
            x_pocket, one_hot_pocket = utils.residues_to_atoms(
                x_pocket, self.dataset_info)

        x = torch.cat((x_lig, x_pocket), dim=1)
        one_hot = torch.cat((one_hot_lig, one_hot_pocket), dim=1)

        # flatten (treat frame (chain dimension) as batch for visualization)
        x_flat = x.view(-1, x.size(-1))
        one_hot_flat = one_hot.view(-1, one_hot.size(-1))
        mask_flat = torch.arange(x.size(0)).repeat_interleave(x.size(1))

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}', 'chain')
        save_xyz_file(str(outdir), one_hot_flat, x_flat, self.dataset_info,
                      name='/chain', batch_mask=mask_flat)
        visualize_chain(str(outdir), self.dataset_info, wandb=wandb)

    @torch.no_grad()
    def generate_ligands(self, pdb_file, n_samples, pocket_ids=None,
                         ref_ligand=None, num_nodes_lig=None, sanitize=False,
                         largest_frag=False, relax_iter=0, timesteps=None,
                         **kwargs):
        """
        Generate ligands given a pocket
        Args:
            pdb_file: PDB filename
            n_samples: number of samples
            pocket_ids: list of pocket residues in <chain>:<resi> format
            ref_ligand: alternative way of defining the pocket based on a
                reference ligand given in <chain>:<resi> format
            num_nodes_lig: number of ligand nodes for each sample (list of
                integers), sampled randomly if 'None'
            sanitize: whether to sanitize molecules or not
            largest_frag: only return the largest fragment
            relax_iter: number of force field optimization steps
            timesteps: number of denoising steps, use training value if None
            kwargs: additional inpainting parameters
        Returns:
            list of molecules
        """

        assert (pocket_ids is None) ^ (ref_ligand is None)
        # print('n_samples',n_samples)
        # Load PDB
        print(pdb_file)
        pdb_struct = PDBParser(QUIET=True).get_structure('', pdb_file)[0]
        
        if pocket_ids is not None:
            # define pocket with list of residues
            residues = [
                pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]
                for x in pocket_ids]

        else:
            # define pocket with reference ligand
            residues = utils.get_pocket_from_ligand(pdb_struct, ref_ligand)
        #print(residues)
        pocket_coord_full = []
        pocket_one_hot_full = []
        if self.pocket_representation == 'CA':
            pocket_coord = torch.tensor(np.array(
                [res['CA'].get_coord() for res in residues]),
                device=self.device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(
                [self.pocket_type_encoder[three_to_one(res.get_resname())]
                 for res in residues], device=self.device)
            
            for res in residues:
                for atom in res.get_atoms():
                    if atom.name == 'CA':
                        continue
                    else:
                        pocket_one_hot_full.append(np.eye(1,len(self.pocket_type_encoder),
                                                   self.pocket_type_encoder[three_to_one(res.get_resname())]).squeeze())
                        pocket_coord_full.append(atom.coord)
               
        else:
            pocket_atoms = [a for res in residues for a in res.get_atoms()
                            if (a.element.capitalize() in self.pocket_type_encoder or a.element != 'H')]
            pocket_coord = torch.tensor(np.array(
                [a.get_coord() for a in pocket_atoms]),
                device=self.device, dtype=FLOAT_TYPE)
            pocket_types = torch.tensor(
                [self.pocket_type_encoder[a.element.capitalize()]
                 for a in pocket_atoms], device=self.device)

        pocket_one_hot = F.one_hot(
            pocket_types, num_classes=len(self.pocket_type_encoder)
        )

        pocket_size = torch.tensor([len(pocket_coord)] * n_samples,
                                   device=self.device, dtype=INT_TYPE)
        pocket_mask = torch.repeat_interleave(
            torch.arange(n_samples, device=self.device, dtype=INT_TYPE),
            len(pocket_coord)
        )

        pocket_one_hot_full = torch.tensor(np.array(pocket_one_hot_full),device=self.device)

        pocket_coord_full = torch.tensor(np.array(pocket_coord_full),device=self.device)
        pocket_size_full = torch.tensor([len(pocket_coord_full)] * n_samples,
                                        device=self.device, dtype=INT_TYPE)

        pocket_mask_full = torch.repeat_interleave(
            torch.arange(n_samples, device=self.device, dtype=INT_TYPE),
            len(pocket_coord_full)
        )
           
        pocket = {
            'x': pocket_coord.repeat(n_samples, 1),
            'x_full':pocket_coord_full.repeat(n_samples,1),
            'one_hot': pocket_one_hot.repeat(n_samples, 1),
            'size': pocket_size,
            'mask': pocket_mask,
            'one_hot_full': pocket_one_hot_full.repeat(n_samples, 1),
            'mask_full': pocket_mask_full,
            'size_full':pocket_size_full
        }

        x_residues = pocket['x'][:, :3].clone()

        cluster_indices1, cluster_indices2 = self.apart_data(x_residues, pocket['mask'])


        pocket1, pocket2 = self.split_pocket_by_index_generate(pocket, cluster_indices1[0], cluster_indices2[0], len(cluster_indices1[0]), len(cluster_indices2[0]),n_samples)
        print(pocket1['size'],pocket2['size'])
        pockets = [pocket1,pocket2]
        pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        # Create dummy ligands
        # Pocket's center of mass
        # print(pockets)
        # pocket = self.merge_pocket(pocket1,pocket2)
        xh_lig_new = []
        xh_pocket_new = []
        lig_mask_new = []
        pocket_mask_new = []
        for i in pockets:
            print(i['size'])
            if num_nodes_lig is None:
                num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
                    n1=None, n2=i['size'])
            # print(num_nodes_lig)
            num_nodes_lig = torch.ceil(num_nodes_lig/2).int()
            print('num_nodes_lig',num_nodes_lig)
            if type(self.ddpm) == ConditionalDDPM:
                xh_lig, xh_pocket, lig_mask, pocket_mask = \
                    self.ddpm.sample_given_pocket(i, num_nodes_lig,
                                                  timesteps=timesteps)
                xh_lig_new.append(xh_lig)
                xh_pocket_new.append(xh_pocket)
                lig_mask_new.append(lig_mask)
                pocket_mask_new.append(pocket_mask)
            else:
                    raise NotImplementedError
        xh_pocket = torch.cat((xh_pocket_new[0],xh_pocket_new[1]),dim=0)
        xh_lig = torch.cat((xh_lig_new[0],xh_lig_new[1]),dim=0)
        lig_mask = torch.cat((lig_mask_new[0],lig_mask_new[1]),dim=0)
        pocket_mask = torch.cat((pocket_mask_new[0],pocket_mask_new[1]),dim=0)

        # Move generated molecule back to the original pocket position
        pocket_com_after = scatter_mean(
            xh_pocket[:, :self.x_dims], pocket_mask, dim=0)

        xh_pocket[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_lig[:, :self.x_dims] += \
            (pocket_com_before - pocket_com_after)[lig_mask]

        # Build mol objects
        x = xh_lig[:, :self.x_dims].detach().cpu()
        atom_type = xh_lig[:, self.x_dims:].argmax(1).detach().cpu()

        molecules = []
        for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                          utils.batch_to_list(atom_type, lig_mask)):

            mol = build_molecule(*mol_pc, self.dataset_info, add_coords=True)
            mol = process_molecule(mol,
                                   add_hydrogens=False,
                                   sanitize=sanitize,
                                   relax_iter=relax_iter,
                                   largest_frag=largest_frag)
            if mol is not None:
                molecules.append(mol)

        return molecules

    def split_pocket_by_index_generate(self,pocket, result1,result2,len11,len21,n_samples):
        pocket1 = {}
        pocket2 = {}
        # print('len11,len21',len11,len21)
        for s in {'x','one_hot'}:
            pocket1[s] = pocket[s][result1].repeat(n_samples, 1)
            pocket2[s] = pocket[s][result2].repeat(n_samples, 1)
        pocket1['mask'] = torch.repeat_interleave(
            torch.arange(n_samples, device=self.device, dtype=INT_TYPE),
            len11
        )
        pocket2['mask'] = torch.repeat_interleave(
            torch.arange(n_samples, device=self.device, dtype=INT_TYPE),
            len21
        )
        pocket1['size'] = torch.tensor([len11] * n_samples,
                     device=self.device, dtype=INT_TYPE)
        pocket2['size'] = torch.tensor([len21] * n_samples,
                                       device=self.device, dtype=INT_TYPE)

        for key in {'x_full','one_hot_full','mask_full','size_full'}:
            pocket1[key] = pocket[key]
            pocket2[key] = pocket[key]

        return pocket1,pocket2
    # 梯度优化
    def configure_gradient_clipping(self, optimizer, optimizer_idx,
                                    gradient_clip_val, gradient_clip_algorithm):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
                        2 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')



