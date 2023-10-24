import math
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

import utils
from torch import nn

class ConditionalDDPM(nn.Module):
    def __init__(
            self,
            dynamics: nn.Module, atom_nf: int, residue_nf: int,
            n_dims: int, size_histogram: Dict,
            timesteps: int = 1000, parametrization='eps',
            noise_schedule='learned', noise_precision=1e-4,
            loss_type='vlb', norm_values=(1., 1.), norm_biases=(None, 0.)):
        super().__init__()

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type

        # Only supported parametrization.
        assert parametrization == 'eps'

        self.gamma = PredefinedNoiseSchedule(noise_schedule,
                                                 timesteps=timesteps,
                                                 precision=noise_precision)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.atom_nf = atom_nf
        self.residue_nf = residue_nf
        self.n_dims = n_dims
        self.num_classes = self.atom_nf

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

        #  distribution of nodes
        self.size_distribution = DistributionNodes(size_histogram)

        # if noise_schedule != 'learned':
        #     self.check_issues_norm_values()
    # 计算KL的
    def kl_prior(self, xh_lig, mask_lig, num_nodes):
        batch_size = len(num_nodes)

        # Compute the last alpha value, alpha_T.
        ones = torch.ones((batch_size, 1), device=xh_lig.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh_lig)

        # Compute means.
        mu_T_lig = alpha_T[mask_lig] * xh_lig
        mu_T_lig_x, mu_T_lig_h = \
            mu_T_lig[:, :self.n_dims], mu_T_lig[:, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_lig_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_lig_h).squeeze()

        # Compute KL for h-part.
        zeros = torch.zeros_like(mu_T_lig_h)
        ones = torch.ones_like(sigma_T_h)
        mu_norm2 = self.sum_except_batch((mu_T_lig_h - zeros) ** 2, mask_lig)
        kl_distance_h = self.gaussian_KL(mu_norm2, sigma_T_h, ones, d=1)

        # Compute KL for x-part.
        zeros = torch.zeros_like(mu_T_lig_x)
        ones = torch.ones_like(sigma_T_x)
        mu_norm2 = self.sum_except_batch((mu_T_lig_x - zeros) ** 2, mask_lig)
        subspace_d = self.subspace_dimensionality(num_nodes)
        kl_distance_x = self.gaussian_KL(mu_norm2, sigma_T_x, ones, subspace_d)

        return kl_distance_x + kl_distance_h

    def subspace_dimensionality(self, input_size):
        """Compute the dimensionality on translation-invariant linear subspace
        where distributions on x are defined."""
        return (input_size - 1) * self.n_dims
    def log_pxh_given_z0_without_constants(self, ligand, z_0_lig, eps_lig,
                                           net_out_lig, gamma_0, epsilon=1e-10):

        # Discrete properties are predicted directly from z_t.
        z_h_lig = z_0_lig[:, self.n_dims:]

        # Take only part over x.
        eps_lig_x = eps_lig[:, :self.n_dims]
        net_lig_x = net_out_lig[:, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0_lig)
        sigma_0_cat = sigma_0 * self.norm_values[1]

        # Computes the error for the distribution
        # N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z0_without_constants_ligand = -0.5 * (
            self.sum_except_batch((eps_lig_x - net_lig_x) ** 2, ligand['mask'])
        )

        # Compute delta indicator masks.
        # un-normalize
        ligand_onehot = ligand['one_hot'] * self.norm_values[1] + self.norm_biases[1]

        estimated_ligand_onehot = z_h_lig * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded.
        centered_ligand_onehot = estimated_ligand_onehot - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional_ligand = torch.log(
            self.cdf_standard_gaussian((centered_ligand_onehot + 0.5) / sigma_0_cat[ligand['mask']])
            - self.cdf_standard_gaussian((centered_ligand_onehot - 0.5) / sigma_0_cat[ligand['mask']])
            + epsilon
        )

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional_ligand, dim=1,
                                keepdim=True)
        log_probabilities_ligand = log_ph_cat_proportional_ligand - log_Z

        # Select the log_prob of the current category using the onehot
        # representation.
        log_ph_given_z0_ligand = self.sum_except_batch(
            log_probabilities_ligand * ligand_onehot, ligand['mask'])

        return log_p_x_given_z0_without_constants_ligand, log_ph_given_z0_ligand

    def sample_p_xh_given_z0(self, z0_lig, xh0_pocket,xh0_pocket_full, lig_mask, pocket_mask,pocket_mask_full,
                             batch_size, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = self.gamma(t_zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0)
        net_out_lig, _ = self.dynamics(
            z0_lig, xh0_pocket,xh0_pocket_full, t_zeros, lig_mask, pocket_mask,pocket_mask_full)

        # Compute mu for p(zs | zt).
        mu_x_lig = self.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
        xh_lig, xh0_pocket,xh0_pocket_full = self.sample_normal_zero_com(
            mu_x_lig, xh0_pocket,xh0_pocket_full, sigma_x, lig_mask, pocket_mask,pocket_mask_full, fix_noise)

        x_lig, h_lig = self.unnormalize(
            xh_lig[:, :self.n_dims], z0_lig[:, self.n_dims:])
        x_pocket, h_pocket = self.unnormalize(
            xh0_pocket[:, :self.n_dims], xh0_pocket[:, self.n_dims:])

        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)
        # h_pocket = F.one_hot(torch.argmax(h_pocket, dim=1), self.residue_nf)

        return x_lig, h_lig, x_pocket, h_pocket

    def sample_normal(self, *args):
        raise NotImplementedError("Has been replaced by sample_normal_zero_com()")
    # 对ligand加噪，并对加噪后的结果进行归一化
    def sample_normal_zero_com(self, mu_lig, xh0_pocket,xh0_pocket_full, sigma, lig_mask,
                               pocket_mask,pocket_mask_full, fix_noise=False):
        """Samples from a Normal distribution."""
        if fix_noise:
            # bs = 1 if fix_noise else mu.size(0)
            raise NotImplementedError("fix_noise option isn't implemented yet")

        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf),
            device=lig_mask.device)

        out_lig = mu_lig + sigma[lig_mask] * eps_lig
        temp = out_lig
        # project to COM-free subspace
        xh_pocket = xh0_pocket.detach().clone()
        xh_pocket_full = xh0_pocket_full.detach().clone()
        out_lig[:, :self.n_dims], xh_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(out_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   lig_mask, pocket_mask)

        temp[:, :self.n_dims],xh_pocket_full[:, :self.n_dims] = \
            self.remove_mean_batch(temp[:, :self.n_dims],
                                   xh_pocket_full[:, :self.n_dims],
                                   lig_mask, pocket_mask_full)
        return out_lig, xh_pocket,xh_pocket_full
    # 前向加噪过程（仅仅只是前向过程中某个时刻的Zt)

    def noised_representation(self, xh_lig, xh0_pocket,xh0_pocket_full,lig_mask, pocket_mask,pocket_mask_full,gamma_t):
        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, xh_lig)
        sigma_t = self.sigma(gamma_t, xh_lig)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf),
            device=lig_mask.device)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t_lig = alpha_t[lig_mask] * xh_lig + sigma_t[lig_mask] * eps_lig

        # project to COM-free subspace
        temp = z_t_lig
        xh_pocket = xh0_pocket.detach().clone()
        xh_pocket_full = xh0_pocket_full.detach().clone()
        z_t_lig[:, :self.n_dims], xh_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(z_t_lig[:, :self.n_dims],
                                   xh_pocket[:, :self.n_dims],
                                   lig_mask, pocket_mask)
        temp[:, :self.n_dims],xh_pocket_full[:, :self.n_dims] = \
            self.remove_mean_batch(temp[:, :self.n_dims],
                                   xh_pocket_full[:, :self.n_dims],
                                   lig_mask, pocket_mask_full)
        return z_t_lig, xh_pocket,xh_pocket_full,eps_lig
    # 此处计算的是pocket的原子个数，ligand的原子个数的关联关系
    def log_pN(self, N_lig, N_pocket):

        log_pN = self.size_distribution.log_prob_n1_given_n2(N_lig, N_pocket)
        return log_pN

    def delta_log_px(self, num_nodes):
        return -self.subspace_dimensionality(num_nodes) * \
               np.log(self.norm_values[0])

    def forward(self,ligand1,pocket1,ligand2,pocket2, return_info=False):
        """
        Computes the loss and NLL terms
        """
        ligand_before = self.merge_ligand(ligand1, ligand2)
        # Likelihood change due to normalization
        delta_log_px = self.delta_log_px(ligand_before['size'])
        # print(ligand_before['size'],ligand1['size'],ligand2['size'])
        ligand_apart = [ligand1,ligand2]
        pocket_apart = [pocket1,pocket2]
        x0_apart = []

        net_out_lig_new = []
        error_t_lig_new = []
        SNR_weight_new = []
        loss_0_x_ligand_new = []
        loss_0_h_new = []
        neg_log_constants_new = []
        kl_prior_new = []
        t_int_new = []
        xh_lig_hat_new = []
        for i in range(len(ligand_apart)):
            # Normalize data, take into account volume change in x.
            ligand, pocket = self.normalize(ligand_apart[i], pocket_apart[i])

            # Sample a timestep t for each example in batch
            # At evaluation time, loss_0 will be computed separately to decrease
            # variance in the estimator (costs two forward passes)
            lowest_t = 0 if self.training else 1
            t_int = torch.randint(
                lowest_t, self.T + 1, size=(ligand['size'].size(0), 1),
                device=ligand['x'].device).float()
            s_int = t_int - 1  # previous timestep

            # Masks: important to compute log p(x | z0).
            t_is_zero = (t_int == 0).float()
            t_is_not_zero = 1 - t_is_zero

            # Normalize t to [0, 1]. Note that the negative
            # step of s will never be used, since then p(x | z0) is computed.
            s = s_int / self.T
            t = t_int / self.T


            gamma_s = self.inflate_batch_array(self.gamma(s), ligand['x'])
            gamma_t = self.inflate_batch_array(self.gamma(t), ligand['x'])

            # Concatenate x, and h[categorical].
            xh0_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
            xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
            xh0_pocket_full = torch.cat([pocket['x_full'],pocket['one_hot_full']],dim=1)
            # Center the input nodes
            temp = xh0_lig
            xh0_lig[:, :self.n_dims], xh0_pocket[:, :self.n_dims] = \
                self.remove_mean_batch(xh0_lig[:, :self.n_dims],
                                       xh0_pocket[:, :self.n_dims],
                                       ligand['mask'], pocket['mask'])
            temp[:, :self.n_dims],xh0_pocket_full[:, :self.n_dims] = \
                self.remove_mean_batch(temp[:, :self.n_dims],
                                       xh0_pocket_full[:, :self.n_dims],
                                       ligand['mask'],pocket['mask_full'])
            # Find noised representation
            z_t_lig, xh_pocket,xh_pocket_full,eps_t_lig = \
                self.noised_representation(xh0_lig, xh0_pocket,xh0_pocket_full, ligand['mask'],
                                           pocket['mask'], pocket['mask_full'],gamma_t)

            # Neural net prediction.
            net_out_lig, _ = self.dynamics(
                z_t_lig, xh_pocket,xh_pocket_full, t, ligand['mask'], pocket['mask'],pocket['mask_full'])
            net_out_lig_new.append(net_out_lig)
            # For LJ loss term
            # xh_lig_hat does not need to be zero-centered as it is only used for
            # computing relative distances
            xh_lig_hat = self.xh_given_zt_and_epsilon(z_t_lig, net_out_lig, gamma_t,
                                                      ligand['mask'])

            # Compute the L2 error.（预测的是噪音的损失）
            error_t_lig = self.sum_except_batch((eps_t_lig - net_out_lig) ** 2,
                                                ligand['mask'])

            # Compute weighting with SNR: (1 - SNR(s-t)) for epsilon parametrization
            SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)
            assert error_t_lig.size() == SNR_weight.size()


            # The _constants_ depending on sigma_0 from the
            # cross entropy term E_q(z0 | x) [log p(x | z0)].
            neg_log_constants = -self.log_constants_p_x_given_z0(
                n_nodes=ligand['size'], device=error_t_lig.device)

            # The KL between q(zT | x) and p(zT) = Normal(0, 1).
            # Should be close to zero.
            kl_prior = self.kl_prior(xh0_lig, ligand['mask'], ligand['size'])

            if self.training:
                # Computes the L_0 term (even if gamma_t is not actually gamma_0)
                # and this will later be selected via masking.
                log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                    self.log_pxh_given_z0_without_constants(
                        ligand, z_t_lig, eps_t_lig, net_out_lig, gamma_t)

                x0_apart.append(z_t_lig)

                loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand * \
                                  t_is_zero.squeeze()
                loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()

                # apply t_is_zero mask
                error_t_lig = error_t_lig * t_is_not_zero.squeeze()

            else:
                # Compute noise values for t = 0.
                t_zeros = torch.zeros_like(s)
                gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand['x'])

                # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
                z_0_lig, xh_pocket,xh_pocket_full,eps_0_lig = \
                    self.noised_representation(xh0_lig, xh0_pocket,xh0_pocket_full, ligand['mask'],
                                               pocket['mask'],pocket['mask_full'],gamma_0)
                x0_apart.append(z_0_lig)
                net_out_0_lig, _ = self.dynamics(
                    z_0_lig, xh_pocket,xh_pocket_full, t_zeros, ligand['mask'], pocket['mask'],pocket['mask_full'])

                log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                    self.log_pxh_given_z0_without_constants(
                        ligand, z_0_lig, eps_0_lig, net_out_0_lig, gamma_0)
                loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand
                loss_0_h = -log_ph_given_z0

            error_t_lig_new.append(error_t_lig)
            SNR_weight_new.append(SNR_weight)
            loss_0_x_ligand_new.append(loss_0_x_ligand)
            loss_0_h_new.append(loss_0_h)
            neg_log_constants_new.append(neg_log_constants)
            kl_prior_new.append(kl_prior)
            t_int_new.append(t_int)
            xh_lig_hat_new.append(xh_lig_hat)


        ligand_new = self.merge_ligand_by_rdkit(x0_apart[0],x0_apart[1])
        ligand_new_mask = torch.cat((ligand1['mask'],ligand2['mask']),dim=0)
        loss_0_for_new = self.loss_0_x_h(ligand_before,ligand_new,ligand_new_mask)

        # sample size prior
        log_pN = self.log_pN(ligand_before['size'], pocket['size'])

        net_out_lig = torch.cat((net_out_lig_new[0],net_out_lig_new[1]),dim =0)

        info = {
            'eps_hat_lig_x': scatter_mean(
                net_out_lig[:, :self.n_dims].abs().mean(1), ligand_before['mask'],
                dim=0).mean(),
            'eps_hat_lig_h': scatter_mean(
                net_out_lig[:, self.n_dims:].abs().mean(1), ligand_before['mask'],
                dim=0).mean(),
        }

        loss_terms = (delta_log_px, error_t_lig_new[0]+error_t_lig_new[1], torch.tensor(0.0), SNR_weight_new[0]+SNR_weight_new[1],
                      loss_0_x_ligand_new[0]+loss_0_x_ligand_new[1], torch.tensor(0.0), loss_0_h_new[0]+loss_0_h_new[1],
                      neg_log_constants_new[0]+neg_log_constants_new[1], kl_prior_new[0]+kl_prior_new[1], log_pN,
                      torch.cat((t_int_new[0],t_int_new[1]),dim=0).squeeze(), torch.cat((xh_lig_hat_new[0],xh_lig_hat_new[1]),dim=0),loss_0_for_new)

        # loss_terms = (delta_log_px, error_t_lig, torch.tensor(0.0), SNR_weight,
        #               loss_0_x_ligand, torch.tensor(0.0), loss_0_h,
        #               neg_log_constants, kl_prior, log_pN,
        #               t_int.squeeze(), xh_lig_hat,loss_0_for_new)
        return (*loss_terms, info) if return_info else loss_terms

    def euclidean_distance(self,tensor1, tensor2,mask):
        # 计算坐标差的平方
        diff = tensor1 - tensor2
        squared_diff = diff ** 2
        # 按照坐标轴求和
        summed_squared_diff = torch.sum(squared_diff, dim=1)
        # 开根号得到欧氏距离
        euclidean_dist = torch.sqrt(summed_squared_diff)
        unique_values = torch.unique(mask)  # 获取mask张量中的唯一值
        group_sums = torch.zeros(len(unique_values))  # 用于存储每组数据的求和结果

        for i, value in enumerate(unique_values):
            group_dist = euclidean_dist[mask == value]  # 获取属于当前唯一值的组的欧氏距离
            group_sum = group_dist.sum()  # 计算当前组的求和结果
            group_sums[i] = group_sum
        return group_sum
    def cross_entropy(self,ligand1,ligand2,mask):

        # 使用unique函数获取不同的组
        unique_groups = torch.unique(mask)

        # 计算每个组的交叉熵损失
        losses = []
        for group in unique_groups:
            indices = (mask == group).nonzero().squeeze()
            group_ligand1 = ligand1[indices]
            group_ligand2 = ligand2[indices]
            loss = F.cross_entropy(group_ligand2, group_ligand1)
            losses.append(loss)
        loss_h = torch.tensor(losses).cuda()

        return loss_h
    def loss_0_x_h(self,ligand,ligand_new,ligand_new_mask):
        loss_x = self.euclidean_distance(ligand['x'],ligand_new[:,:3],ligand_new_mask)

        # ligand_new = ligand_new[:, 3:].argmax(1).detach()
        ligand = ligand['one_hot'].argmax(1).detach()
        loss_h = self.cross_entropy(ligand,ligand_new[:, 3:],ligand_new_mask)

        loss_0_for_new = loss_x+loss_h
        loss_0_for_new.requires_grad = True
        return loss_0_for_new
    def merge_ligand(self,ligand1,ligand2):
        ligand = {}
        for item in {'x','one_hot','mask','size'}:
            if item == 'size':
                ligand[item] = ligand1[item]+ligand2[item]
                continue
            ligand[item] = torch.cat((ligand1[item],ligand2[item]),dim=0)
        return ligand
    def merge_ligand_by_rdkit(self,ligand1,ligand2):
        ligand_new = torch.cat((ligand1,ligand2))
        return ligand_new
    # 参数重整化--通过预测噪音获取Z（date）的过程
    def xh_given_zt_and_epsilon(self, z_t, epsilon, gamma_t, batch_mask):
        """ Equation (7) in the EDM paper """
        alpha_t = self.alpha(gamma_t, z_t)
        sigma_t = self.sigma(gamma_t, z_t)
        xh = z_t / alpha_t[batch_mask] - epsilon * sigma_t[batch_mask] / \
             alpha_t[batch_mask]
        return xh
    # en_diffusion调用的函数，不确定这里是不是有用
    def sample_p_zt_given_zs(self, zs_lig, xh0_pocket, ligand_mask, pocket_mask,
                             gamma_t, gamma_s, fix_noise=False):
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zs_lig)

        mu_lig = alpha_t_given_s[ligand_mask] * zs_lig
        zt_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma_t_given_s, ligand_mask, pocket_mask,
            fix_noise)

        return zt_lig, xh0_pocket
    # 去噪函数调用的用于计算每一步去噪的值
    def sample_p_zs_given_zt(self,s,t,zt_lig,xh0_pocket,xh0_pocket_full,ligand_mask,pocket_mask,pocket_mask_full,fix_noise=False):
    #     """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
    
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
             self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)
    
        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)
    
         # Neural net prediction.
        eps_t_lig, _ = self.dynamics(
             zt_lig, xh0_pocket,xh0_pocket_full, t, ligand_mask, pocket_mask,pocket_mask_full)
    
         # Compute mu for p(zs | zt).
         # Note: mu_{t->s} = 1 / alpha_{t|s} z_t - sigma_{t|s}^2 / sigma_t / alpha_{t|s} epsilon
         # follows from the definition of mu_{t->s} and Equ. (7) in the EDM paper
        mu_lig = zt_lig / alpha_t_given_s[ligand_mask] - \
                  (sigma2_t_given_s / alpha_t_given_s / sigma_t)[ligand_mask] * \
                  eps_t_lig
    
         # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t
    
         # Sample zs given the parameters derived from zt.
        zs_lig, xh0_pocket,xh0_pocket_full= self.sample_normal_zero_com(
             mu_lig, xh0_pocket,xh0_pocket_full, sigma, ligand_mask, pocket_mask,pocket_mask_full, fix_noise)
    
        self.assert_mean_zero_with_mask(zt_lig[:, :self.n_dims], ligand_mask)
    
        return zs_lig, xh0_pocket,xh0_pocket_full

    def sample_p_zs_given_zt1(self,s,t,zt_lig,xh0_pocket,ligand_mask,pocket_mask,fix_noise=False):
        
        r1 = 0.5
        print('here!')
        print('t',s,t)
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)
        alpha_t = -F.logsigmoid(-gamma_t)
        alpha_s = -F.logsigmoid(-gamma_s)
        print(alpha_t)
        print(sigma_s)
        #t denotes i
        #s denotes i-1
        
        s_lam = torch.div(alpha_s,sigma_s)
        t_lam = torch.div(alpha_t,sigma_t)
        print(s_lam,t_lam)
        lambda_s = torch.log(s_lam)
        lambda_t = torch.log(t_lam)
        print('result:',lambda_s,lambda_t)
        hi = lambda_s - lambda_t
        si = torch.exp(r1*hi + lambda_s)
        #si = si.int()
        print('si:',si)
        gamma_si = self.gamma(si)
        alpha_si = F.logsigmoid(-gamma_si)
        
        sigma2,_ ,_ = self.sigma_and_alpha_t_given_s(gamma_t,gamma_s,zt_lig) 
        eps_t_lig, _ = self.dynamics(
                zt_lig, xh0_pocket, t, ligand_mask, pocket_mask)


        ui = (alpha_si/alpha_t)*zt_lig-gamma_si*(torch.exp(r1*hi)-1)*eps_t_lig

        eps_si_lig,_ = self.dynamics(
                ui,xh0_pocket,si,ligand_mask,pocket_mask)

        zs_lig = (alpha_s/alpha_t)*zt_lig-gamma_t*(torch.exp(hi-1))*eps_t_lig-sigma_s/2/r1*(eps_si_lig-eps_t_lig)

        return zs_lig,xh0_pocket

    def sample_combined_position_feature_noise(self, lig_indices, xh0_pocket,
                                               pocket_indices):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise
        for z_h.
        """
        raise NotImplementedError("Use sample_normal_zero_com() instead.")

    def sample(self, *args):
        raise NotImplementedError("Conditional model does not support sampling "
                                  "without given pocket.")
    # 验证和测试的过程
    #@torch.no_grad()
    def sample_given_pocket(self, pocket, num_nodes_lig, return_frames=1,
                            timesteps=None):
        """
        Draw samples from the generative model. Optionally, return intermediate
        states for visualization purposes.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0
        # print(pocket)
        n_samples = len(pocket['size'])
        device = pocket['x'].device

        _, pocket = self.normalize(pocket=pocket)

        # xh0_pocket is the original pocket while xh_pocket might be a
        # translated version of it
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        xh0_pocket_full = torch.cat([pocket['x_full'],pocket['one_hot_full']],dim=1)

        lig_mask = utils.num_nodes_to_batch_mask(
            n_samples, num_nodes_lig, device)

        # Sample from Normal distribution in the pocket center
        mu_lig_x = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
        # 此处残差是自定义的
        sigma = torch.ones_like(pocket['size']).unsqueeze(1)

        z_lig, xh_pocket,xh_pocket_full = self.sample_normal_zero_com(
            mu_lig, xh0_pocket,xh0_pocket_full, sigma, lig_mask, pocket['mask'],pocket['mask_full'])

        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)
        # 采样部分：采样的大小是根据之前判定好的分布，根据pocket_size选取合适的配体大小
        out_lig = torch.zeros((return_frames,) + z_lig.size(),
                              device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(),
                                 device=device)
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s,
                                 device=z_lig.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            z_lig, xh_pocket,xh_pocket_full = self.sample_p_zs_given_zt(
                s_array, t_array, z_lig, xh_pocket,xh_pocket_full,lig_mask,pocket['mask'],pocket['mask_full'])

            # save frame
            if (s * return_frames) % timesteps == 0:
                idx = (s * return_frames) // timesteps
                out_lig[idx], out_pocket[idx] = \
                    self.unnormalize_z(z_lig, xh_pocket)

        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket,xh_pocket_full,lig_mask, pocket['mask'],pocket['mask_full'], n_samples)

        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        # Correct CoM drift for examples without intermediate states
        if return_frames == 1:
            max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting '
                      f'the positions down.')
                x_lig, x_pocket = self.remove_mean_batch(
                    x_lig, x_pocket, lig_mask, pocket['mask'])

        # Overwrite last frame with the resulting x and h.
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        # remove frame dimension if only the final molecule is returned
        return out_lig.squeeze(0), out_pocket.squeeze(0), lig_mask, \
               pocket['mask']

    @classmethod
    def remove_mean_batch(cls, x_lig, x_pocket, lig_indices, pocket_indices):

        # Just subtract the center of mass of the sampled part
        mean = scatter_mean(x_lig, lig_indices, dim=0)
        
        x_lig = x_lig - mean[lig_indices]
        x_pocket = x_pocket - mean[pocket_indices]
        return x_lig, x_pocket
    def normalize(self, ligand=None, pocket=None):
        if ligand is not None:
            ligand['x'] = ligand['x'] / self.norm_values[0]

            # Casting to float in case h still has long or int type.
            ligand['one_hot'] = \
                (ligand['one_hot'].float() - self.norm_biases[1]) / \
                self.norm_values[1]

        if pocket is not None:
            pocket['x'] = pocket['x'] / self.norm_values[0]
            pocket['one_hot'] = \
                (pocket['one_hot'].float() - self.norm_biases[1]) / \
                self.norm_values[1]

        return ligand, pocket

    @staticmethod
    def inflate_batch_array(array, target):
        """
        Inflates the batch array (array) with only a single axis
        (i.e. shape = (batch_size,), or possibly more empty axes
        (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)
    def unnormalize(self, x, h_cat):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]

        return x, h_cat

    def unnormalize_z(self, z_lig, z_pocket):
        # Parse from z
        x_lig, h_lig = z_lig[:, :self.n_dims], z_lig[:, self.n_dims:]
        x_pocket, h_pocket = z_pocket[:, :self.n_dims], z_pocket[:, self.n_dims:]

        # Unnormalize
        x_lig, h_lig = self.unnormalize(x_lig, h_lig)
        x_pocket, h_pocket = self.unnormalize(x_pocket, h_pocket)
        return torch.cat([x_lig, h_lig], dim=1), \
               torch.cat([x_pocket, h_pocket], dim=1)

    def subspace_dimensionality(self, input_size):
        """Compute the dimensionality on translation-invariant linear subspace
        where distributions on x are defined."""
        return (input_size - 1) * self.n_dims
    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)),
                                        target_tensor)
    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)),
                                        target_tensor)

    @staticmethod
    def SNR(gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)
    @staticmethod
    def sum_except_batch(x, indices):
        return scatter_add(x.sum(-1), indices, dim=0)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))
    @staticmethod
    def sample_gaussian(size, device):
        # print('size:', size, device)
        x = torch.randn(size, device=device)
        # print('sample_gassus_x:', x.shape)
        return x
    def log_constants_p_x_given_z0(self, n_nodes, device):
        """Computes p(x|z0)."""

        batch_size = len(n_nodes)
        degrees_of_freedom_x = self.subspace_dimensionality(n_nodes)

        zeros = torch.zeros((batch_size, 1), device=device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    @staticmethod
    def gaussian_KL(q_mu_minus_p_mu_squared, q_sigma, p_sigma, d):
        """Computes the KL distance between two normal distributions.
            Args:
                q_mu_minus_p_mu_squared: Squared difference between mean of
                    distribution q and distribution p: ||mu_q - mu_p||^2
                q_sigma: Standard deviation of distribution q.
                p_sigma: Standard deviation of distribution p.
                d: dimension
            Returns:
                The KL distance
            """
        return d * torch.log(p_sigma / q_sigma) + \
            0.5 * (d * q_sigma ** 2 + q_mu_minus_p_mu_squared) / \
            (p_sigma ** 2) - 0.5 * d


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined
    (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        # print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        print('这是PredefinedNoiseSchedule')
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]
def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2

def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1.
    This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2

class DistributionNodes:
    def __init__(self, histogram):

        histogram = torch.tensor(histogram).float()
        histogram = histogram + 1e-3  # for numerical stability

        prob = histogram / histogram.sum()

        self.idx_to_n_nodes = torch.tensor(
            [[(i, j) for j in range(prob.shape[1])] for i in range(prob.shape[0])]
        ).view(-1, 2)

        self.n_nodes_to_idx = {tuple(x.tolist()): i
                               for i, x in enumerate(self.idx_to_n_nodes)}

        self.prob = prob
        self.m = torch.distributions.Categorical(self.prob.view(-1),
                                                 validate_args=True)

        self.n1_given_n2 = \
            [torch.distributions.Categorical(prob[:, j], validate_args=True)
             for j in range(prob.shape[1])]
        self.n2_given_n1 = \
            [torch.distributions.Categorical(prob[i, :], validate_args=True)
             for i in range(prob.shape[0])]

        # entropy = -torch.sum(self.prob.view(-1) * torch.log(self.prob.view(-1) + 1e-30))
        entropy = self.m.entropy()
        print("Entropy of n_nodes: H[N]", entropy.item())

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        num_nodes_lig, num_nodes_pocket = self.idx_to_n_nodes[idx].T
        return num_nodes_lig, num_nodes_pocket

    def sample_conditional(self, n1=None, n2=None):
        assert (n1 is None) ^ (n2 is None), \
            "Exactly one input argument must be None"

        m = self.n1_given_n2 if n2 is not None else self.n2_given_n1
        c = n2 if n2 is not None else n1

        return torch.tensor([m[i].sample() for i in c], device=c.device)

    def log_prob(self, batch_n_nodes_1, batch_n_nodes_2):
        assert len(batch_n_nodes_1.size()) == 1
        assert len(batch_n_nodes_2.size()) == 1

        idx = torch.tensor(
            [self.n_nodes_to_idx[(n1, n2)]
             for n1, n2 in zip(batch_n_nodes_1.tolist(), batch_n_nodes_2.tolist())]
        )

        # log_probs = torch.log(self.prob.view(-1)[idx] + 1e-30)
        log_probs = self.m.log_prob(idx)

        return log_probs.to(batch_n_nodes_1.device)

    def log_prob_n1_given_n2(self, n1, n2):
        assert len(n1.size()) == 1
        assert len(n2.size()) == 1
        log_probs = torch.stack([self.n1_given_n2[c].log_prob(i.cpu())
                                 for i, c in zip(n1, n2)])
        return log_probs.to(n1.device)

    def log_prob_n2_given_n1(self, n2, n1):
        assert len(n2.size()) == 1
        assert len(n1.size()) == 1
        log_probs = torch.stack([self.n2_given_n1[c].log_prob(i.cpu())
                                 for i, c in zip(n2, n1)])
        return log_probs.to(n2.device)



