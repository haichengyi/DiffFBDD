import torch
import torch.nn as nn
from equivariant_diffusion.egnn_new import EGNN, GNN
import numpy as np
from torch_scatter import scatter_add, scatter_mean


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x
remove_mean_batch = remove_mean_batch

class EGNNDynamics(nn.Module):
    def __init__(self, atom_nf, residue_nf,
                 n_dims, joint_nf=16, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics',
                 norm_constant=0, inv_sublayers=2, sin_embedding=False,
                 normalization_factor=100, aggregation_method='sum',
                 update_pocket_coords=True, edge_cutoff=None):
        super().__init__()
        self.mode = mode
        self.edge_cutoff = edge_cutoff

        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, joint_nf)
        )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, atom_nf)
        )

        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, joint_nf)
        )

        self.residue_decoder = nn.Sequential(
            nn.Linear(joint_nf, 2 * residue_nf),
            act_fn,
            nn.Linear(2 * residue_nf, residue_nf)
        )

        if condition_time:
            dynamics_node_nf = joint_nf + 1
        else:
            print('Warning: dynamics model is _not_ conditioned on time.')
            dynamics_node_nf = joint_nf

        if mode == 'egnn_dynamics':
            self.egnn = EGNN(
                in_node_nf=dynamics_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.node_nf = dynamics_node_nf
            self.update_pocket_coords = update_pocket_coords

        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=dynamics_node_nf + n_dims, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=n_dims + dynamics_node_nf,
                device=device, act_fn=act_fn, n_layers=n_layers,
                attention=attention, normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)

        self.device = device
        self.n_dims = n_dims
        self.condition_time = condition_time

    def forward(self, xh_atoms, xh_residues,xh_full, t, mask_atoms, mask_residues,mask_full):
        
        x_atoms = xh_atoms[:, :self.n_dims].clone()
        h_atoms = xh_atoms[:, self.n_dims:].clone()
   
        x_residues = xh_residues[:, :self.n_dims].clone()
        h_residues = xh_residues[:, self.n_dims:].clone()
        
        x_full = xh_full[:, :self.n_dims].clone()
        h_full = xh_full[:, self.n_dims:].clone()
        # embed atom features and residue features in a shared space
        
        h_atoms = self.atom_encoder(h_atoms)
        
        h_residues = self.residue_encoder(h_residues)
        
        h_full = self.residue_encoder(h_full)  
        # combine the two node types
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])
        
        #combine pocket_type
        x_pocket = torch.cat((x_residues,x_full), dim=0)
        h_pocket = torch.cat((h_residues,h_full), dim=0)
        mask_pocket = torch.cat([mask_residues,mask_full])
        
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
                h_time_pocket = torch.empty_like(h_pocket[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[mask]                
                h_time_pocket = t[mask_pocket]
                
                
            h = torch.cat([h, h_time], dim=1)
            h_pocket = torch.cat([h_pocket,h_time_pocket],dim=1)
        # get edges of a complete graph
        edges = self.get_edges(mask, x)

        #get edge of pocket
        self.edge_cutoff = 3
        edges_pocket = self.get_edges(mask_pocket, x_pocket)
        #print(edges_pocket.shape,mask_residues.shape,mask_pocket.shape)
        self.edge_cutoff = None
        #print(self.update_pocket_coords)

        if self.mode == 'egnn_dynamics':
            update_coords_mask = None if self.update_pocket_coords \
                else torch.cat((torch.ones_like(mask_atoms),
                                torch.zeros_like(mask_residues))).unsqueeze(1)


            h_final, x_final = self.egnn(h, x, edges,h_pocket,x_pocket,edges_pocket,mask_residues,
                                         update_coords_mask=update_coords_mask)
            vel = (x_final - x)

        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=None)
            vel = output[:, :3]
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        # decode atom and residue features
        h_final_atoms = self.atom_decoder(h_final[:len(mask_atoms)])
        h_final_residues = self.residue_decoder(h_final[len(mask_atoms):])

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if self.update_pocket_coords:
            # in case of unconditional joint distribution, include this as in
            # the original code
            vel = remove_mean_batch(vel, mask)

        return torch.cat([vel[:len(mask_atoms)], h_final_atoms], dim=-1), \
               torch.cat([vel[len(mask_atoms):], h_final_residues], dim=-1)

    def get_edges(self, batch_mask, x):
        # TODO: cache batches for each example in self._edges_dict[n_nodes]
        adj = batch_mask[:, None] == batch_mask[None, :]
        if self.edge_cutoff is not None:
            adj = adj & (torch.cdist(x, x) <= self.edge_cutoff)
        edges = torch.stack(torch.where(adj), dim=0)
        return edges
