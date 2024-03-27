import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydantic.typing import Literal
from torch_geometric.nn import Linear, MessagePassing, global_mean_pool
from torch_geometric.nn.models.schnet import ShiftedSoftplus

from models.base import BaseSettings
from models.transformer import TransformerConv

from models.utils import RBFExpansion

from correction_terms.electrostatic_energy import ElectrostaticEnergy
from correction_terms.d4_dispersion_energy import D4DispersionEnergy
from correction_terms.zbl_repulsion_energy import ZBLRepulsionEnergy
from ewald_block import EwaldBlock
from ewald_auxiliary.utils import (
    get_k_index_product_set,
    get_k_voxel_grid,
    pos_svd_frame,
    x_to_k_cell,
)
from ewald_auxiliary.base_layers import Dense

class PotNetConfig(BaseSettings):
    name: Literal["potnet"]
    conv_layers: int = 3
    atom_input_features: int = 92
    inf_edge_features: int = 64
    fc_features: int = 256
    output_dim: int = 256
    output_features: int = 1
    rbf_min = -4.0
    rbf_max = 4.0
    potentials = []
    euclidean = False
    charge_map = False
    transformer = False
    

        # Spookynet corrections
    lr_cutoff: float = None
    cutoff: float = 8
    use_zbl_repulsion: bool = False
    use_electrostatics: bool = False
    long_range_only: bool = False
    use_d4_dispersion: bool = False
    compute_d4_atomic: bool = False
    learnable_parameters: bool = False

    # Ewald-Message-Passing
    use_ewald: bool = False
    use_pbc: bool = False
    num_k_x: int = 2                              # Frequency cutoff [Å^-1]
    num_k_y: int = 2                              # Voxel grid resolution [Å^-1]
    num_k_z: int = 2                              # Gaussian radial basis size (Fourier filter)
    downprojection_size: int = 8                  # Size of linear bottleneck layer
    num_hidden: int = 3                           # Number of residuals in update function    
    k_cutoff: float = 0.4                         # Frequency cutoff [Å^-1]
    delta_k: float = 0.2                          # Voxel grid resolution [Å^-1]
    num_k_rbf: int = 1                            # Gaussian radial basis size (Fourier filter)

    class Config:
        """Configure model settings behavior."""
        env_prefix = "jv_model"


class PotNetConv(MessagePassing):

    def __init__(self, fc_features):
        super(PotNetConv, self).__init__(node_dim=0)
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction = nn.BatchNorm1d(fc_features)
        self.nonlinear_full = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )

        return F.relu(x + self.bn(out))

    def message(self, x_i, x_j, edge_attr, index):
        score = torch.sigmoid(self.bn_interaction(self.nonlinear_full(torch.cat((x_i, x_j, edge_attr), dim=1))))
        return score * self.nonlinear(torch.cat((x_i, x_j, edge_attr), dim=1))

class PotNet(nn.Module):

    def __init__(self, config: PotNetConfig = PotNetConfig(name="potnet")):
        super().__init__()
        self.config = config

        # Spookynet corrections
        self.use_zbl_repulsion = config.use_zbl_repulsion
        self.use_electrostatics = config.use_electrostatics
        self.long_range_only = config.long_range_only        
        self.use_d4_dispersion = config.use_d4_dispersion
        self.lr_cutoff = config.lr_cutoff
        self.cutoff = config.cutoff
        self.compute_d4_atomic = config.compute_d4_atomic
        self.learnable_parameters = config.learnable_parameters

        # Ewald parameters
        self.use_ewald = config.use_ewald
        self.use_pbc = config.use_pbc
        self.hidden_channels = config.fc_features
        self.conv_layers = config.conv_layers
        # Parse Ewald hyperparams
        if self.use_ewald:
            # Integer values to define box of k-lattice indices
            if self.use_pbc:
                self.num_k_x = config.num_k_x
                self.num_k_y = config.num_k_y
                self.num_k_z = config.num_k_z
                self.delta_k = None
            else:
                self.k_cutoff = config.k_cutoff
                self.delta_k = config.delta_k
                self.num_k_rbf = config.num_k_rbf
            self.downprojection_size = config.downprojection_size
            # Number of residuals in update function
            self.num_hidden = config.num_hidden

        if not config.charge_map:
            self.atom_embedding = nn.Linear(
                config.atom_input_features, config.fc_features
            )
        else:
            self.atom_embedding = nn.Linear(
                config.atom_input_features + 10, config.fc_features
            )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=config.rbf_min,
                vmax=config.rbf_max,
                bins=config.fc_features,
            ),
            nn.Linear(config.fc_features, config.fc_features),
            nn.SiLU(),
        )

        if not self.config.euclidean:
            self.inf_edge_embedding = RBFExpansion(
                vmin=config.rbf_min,
                vmax=config.rbf_max,
                bins=config.inf_edge_features,
                type='multiquadric'
            )

            self.infinite_linear = nn.Linear(config.inf_edge_features, config.fc_features)

            self.infinite_bn = nn.BatchNorm1d(config.fc_features)

        self.conv_layers = nn.ModuleList(
            [
                PotNetConv(config.fc_features)
                for _ in range(config.conv_layers)
            ]
        )

        if not config.euclidean and config.transformer:
            self.transformer_conv_layers = nn.ModuleList(
                [
                    TransformerConv(config.fc_features, config.fc_features)
                    for _ in range(config.conv_layers)
                ]
            )

        self.fc = nn.Sequential(
            nn.Linear(config.fc_features, config.fc_features), ShiftedSoftplus()
        )

        self.fc_out = nn.Linear(config.output_dim, 2)

        # Scales for correction terms
        self.rescale1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.rescale2 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.rescale3 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.Softplus = nn.Softplus()        
        
        # Load corrections
        if self.use_d4_dispersion:
            self.d4_dispersion_energy = D4DispersionEnergy(cutoff=self.lr_cutoff)

        if self.use_electrostatics:
            self.electrostatic_energy = ElectrostaticEnergy(
                cuton=0.25 * self.cutoff,
                cutoff=0.75 * self.cutoff,
                lr_cutoff=self.lr_cutoff,
                long_range_only=self.long_range_only,                
            )    

        if self.use_zbl_repulsion:
            self.zbl_repulsion_energy = ZBLRepulsionEnergy()

         # Initialize k-space structure (Ewald-Message-Passing)
        if self.use_ewald:
            # Get the reciprocal lattice indices of included k-vectors
            if self.use_pbc:
                (
                    self.k_index_product_set,
                    self.num_k_degrees_of_freedom,
                ) = get_k_index_product_set(
                    self.num_k_x,
                    self.num_k_y,
                    self.num_k_z,
                )
                self.k_rbf_values = None
                self.delta_k = None
            else:
                # Get the k-space voxel and evaluate Gaussian RBF (can be done at
                # initialization time as voxel grid stays fixed for all structures)
                (
                    self.k_grid,
                    self.k_rbf_values,
                    self.num_k_degrees_of_freedom,
                ) = get_k_voxel_grid(
                    self.k_cutoff,
                    self.delta_k,
                    self.num_k_rbf,
                )
            # Downprojection layer, weights are shared among all interaction blocks
            self.down = Dense(
                self.num_k_degrees_of_freedom,
                self.downprojection_size,
                activation=None,
                bias=False,
            )

            self.ewald_blocks = torch.nn.ModuleList(
                [
                    EwaldBlock(
                        self.down,
                        self.hidden_channels,  # Embedding size of short-range GNN
                        self.downprojection_size,
                        self.num_hidden,  # Number of residuals in update function
                        activation="silu",
                        use_pbc=True,
                        delta_k=self.delta_k,
                        k_rbf_values=self.k_rbf_values,
                    )
                    for i in range(config.conv_layers)
                ]
            )

        self.skip_connection_factor = (
            2.0 + float(self.use_ewald)
        ) ** (-0.5)        

    def forward(self, data, print_data=False):
        """CGCNN function mapping graph to outputs."""
        # fixed edge features: RBF-expanded bondlengths

        # Atomic positions
        tensor_list = [torch.tensor(array) for array in data.coords]  # Convert each array to a tensor
        R = torch.cat(tensor_list, dim=0)

        # Nuclear charges (atomic numbers) of atoms
        Z = data.atom_numbers.squeeze()

        # Lattice information
        cell = torch.tensor(data.lattice_mat)

        # Number of atoms
        N = len(data.x)

        # Batch size
        num_batch = torch.max(data.batch).item() + 1

        # Index for each atom that specifies to which molecule in the batch it belongs.
        batch_seg = data.batch

        # Index of atom i for all atomic pairs ij. Each pair must be specified as both ij and ji.
        idx_i = data.edge_index[0]
        # Same as idx_i, but for atom j.
        idx_j = data.edge_index[1]

        # Interatomic distances
        rij = data.edge_attr

        edge_index = data.edge_index
        if self.config.euclidean:
            edge_features = self.edge_embedding(data.edge_attr)
        else:
            edge_features = self.edge_embedding(-0.75 / data.edge_attr)
        
        if not self.config.euclidean:
            inf_edge_index = data.inf_edge_index
            inf_feat = sum([data.inf_edge_attr[:, i] * pot for i, pot in enumerate(self.config.potentials)])
            inf_edge_features = self.inf_edge_embedding(inf_feat)
            inf_edge_features = self.infinite_bn(F.softplus(self.infinite_linear(inf_edge_features)))

        # initial node features: atom feature network...
        if self.config.charge_map:
            node_features = self.atom_embedding(torch.cat([data.x, data.g_feats], -1))
        else:
            node_features = self.atom_embedding(data.x)

        if not self.config.euclidean and not self.config.transformer:
            edge_index = torch.cat([data.edge_index, inf_edge_index], 1)
            edge_features = torch.cat([edge_features, inf_edge_features], 0)

        
        if self.use_ewald:
            if self.use_pbc:
                # Compute reciprocal lattice basis of structure
                k_cell, _ = x_to_k_cell(cell)
                # Translate lattice indices to k-vectors
                k_grid = torch.matmul(
                    self.k_index_product_set.double(), k_cell
                )
                k_grid = k_grid.to(batch_seg.device)
            else:
                k_grid = (
                    self.k_grid.to(batch_seg.device)
                    .unsqueeze(0)
                    .expand(num_batch, -1, -1)
                )
                k_grid = k_grid.to(batch_seg.device)

        for i in range(self.config.conv_layers):
            dot = None  # These will be computed in first Ewald block and then passed
            sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
            if self.use_ewald:
                f_ewald, dot, sinc_damping = self.ewald_blocks[i](
                                node_features,
                                R.to(batch_seg.device),
                                k_grid,
                                num_batch,
                                batch_seg,
                                dot,
                                sinc_damping,
                            )
            if not self.config.euclidean and self.config.transformer:
                local_node_features = self.conv_layers[i](node_features, edge_index, edge_features)
                inf_node_features = self.transformer_conv_layers[i](node_features, inf_edge_index, inf_edge_features)
                node_features = local_node_features + inf_node_features
                if self.use_ewald:
                    node_features = self.skip_connection_factor * (node_features + f_ewald)
            else:
                node_features = self.conv_layers[i](node_features, edge_index, edge_features)
                if self.use_ewald:
                    node_features = self.skip_connection_factor * (node_features + f_ewald)


        # WE need output for both partial charges "qa" and atomic energy "ea"
        features = self.fc(node_features)
        out = self.fc_out(features)

        ea = out.narrow(-1, 0, 1).squeeze(-1)  # atomic energy
        qa = out.narrow(-1, 1, 1).squeeze(-1)  # partial charge

        # Assume total charge Q = 0
        Q = torch.zeros(num_batch, dtype=qa.dtype,device=qa.device)

        # correct partial charges for charge conservation
        # (spread leftover charge evenly over all atoms)
        w = torch.ones(N, dtype=qa.dtype, device=qa.device)
        Qleftover = Q.index_add(0, batch_seg, -qa)
        wnorm = w.new_zeros(num_batch).index_add_(0, batch_seg, w)
        if w.device.type == "cpu":  # indexing is faster on CPUs
            w = w / wnorm[batch_seg]
            qa = qa + w * Qleftover[batch_seg]
        else:  # gathering is faster on GPUs
            w = w / torch.gather(wnorm, 0, batch_seg)
            qa = qa + w * torch.gather(Qleftover, 0, batch_seg)

        # Function used for ZBL-repulsion-energy
        def cutoff_function(x: torch.Tensor, cutoff: float) -> torch.Tensor:
            """
            Cutoff function that smoothly goes from f(x) = 1 to f(x) = 0 in the interval
            from x = 0 to x = cutoff. For x >= cutoff, f(x) = 0. This function has
            infinitely many smooth derivatives. Only positive x should be used as input.
            """
            zeros = torch.zeros_like(x)
            x_ = torch.where(x < cutoff, x, zeros)  # prevent nan in backprop
            return torch.where(
                x < cutoff, torch.exp(-(x_ ** 2) / ((cutoff - x_) * (cutoff + x_))), zeros
            )
        
        # ZBL inspired short-range repulsion
        if self.use_zbl_repulsion:
            cutmask = rij < self.cutoff  # select all entries below cutoff
            sr_rij = rij[cutmask]
            sr_idx_i = idx_i[cutmask]
            sr_idx_j = idx_j[cutmask]
            cutoff_values = cutoff_function(sr_rij, self.cutoff)

            ea_rep = self.zbl_repulsion_energy(
                N, Z.to(torch.float), sr_rij, cutoff_values, sr_idx_i, sr_idx_j
            )
        else:
            ea_rep = ea.new_zeros(N)


        # Compute electrostatic Energy
        if self.use_electrostatics:
            ea_ele = self.electrostatic_energy(
                N, qa, rij, idx_i, idx_j, R, cell, num_batch, batch_seg
            )
        else:
            ea_ele = ea.new_zeros(N)

        # Grimme's D4 dispersion
        if self.use_d4_dispersion:
            ea_vdw, pa, c6 = self.d4_dispersion_energy(
                N, Z, qa, rij, idx_i, idx_j, self.compute_d4_atomic
            )
        else:
            ea_vdw, pa, c6 = ea.new_zeros(N), ea.new_zeros(N), ea.new_zeros(N)

        # Sum all energies
        if(self.learnable_parameters):
            ea = ea + self.rescale1 * ea_rep \
                    + self.rescale2 * ea_ele \
                    + self.rescale3 * ea_vdw

        else:
            ea = ea + ea_rep + ea_ele + ea_vdw    
        
        # Crystal-level readout
        energy = global_mean_pool(ea, data.batch)
        return torch.squeeze(energy)
