"""Implementation based on the template of ALIGNN."""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pydantic.typing import Literal
from torch import nn
from models.utils import RBFExpansion
from models.base import BaseSettings
from models.features import angle_emb_mp
from torch_scatter import scatter
from models.transformer import MatformerConv
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


class MatformerConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["matformer"]
    conv_layers: int = 5
    edge_layers: int = 0
    atom_input_features: int = 92
    edge_features: int = 128
    triplet_input_features: int = 40
    node_features: int = 128
    fc_layers: int = 1
    fc_features: int = 128
    output_features: int = 1
    node_layer_head: int = 4
    edge_layer_head: int = 4
    nn_based: bool = False

    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False

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


class Matformer(nn.Module):
    """att pyg implementation."""

    def __init__(self, config: MatformerConfig = MatformerConfig(name="matformer")):
        print("Config: ", config)
        """Set up att modules."""
        super().__init__()
        # Ewald parameters
        self.use_ewald = config.use_ewald
        self.use_pbc = config.use_pbc
        self.hidden_channels = config.node_features
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

        self.classification = config.classification # not used
        self.use_angle = config.use_angle # not used

        # Correction terms parameters
        self.use_zbl_repulsion = config.use_zbl_repulsion
        self.use_electrostatics = config.use_electrostatics
        self.long_range_only = config.long_range_only
        self.use_d4_dispersion = config.use_d4_dispersion
        self.lr_cutoff = config.lr_cutoff
        self.cutoff = config.cutoff
        self.compute_d4_atomic = config.compute_d4_atomic
        self.learnable_parameters = config.learnable_parameters
        
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features),
        )
        self.angle_lattice = config.angle_lattice
        if self.angle_lattice: ## module not used
            print('use angle lattice')
            self.lattice_rbf = nn.Sequential(
                RBFExpansion(
                    vmin=0,
                    vmax=8.0,
                    bins=config.edge_features,
                ),
                nn.Linear(config.edge_features, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_angle = nn.Sequential(
                RBFExpansion(
                    vmin=-1,
                    vmax=1.0,
                    bins=config.triplet_input_features,
                ),
                nn.Linear(config.triplet_input_features, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_emb = nn.Sequential(
                nn.Linear(config.node_features * 6, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_atom_emb = nn.Sequential(
                nn.Linear(config.node_features * 2, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )


        self.edge_init = nn.Sequential( ## module not used
            nn.Linear(3 * config.node_features, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features)
        )

        self.sbf = angle_emb_mp(num_spherical=3, num_radial=40, cutoff=8.0) ## module not used

        self.angle_init_layers = nn.Sequential( ## module not used
            nn.Linear(120, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features)
        )

        self.att_layers = nn.ModuleList(
            [
                MatformerConv(in_channels=config.node_features, out_channels=config.node_features, heads=config.node_layer_head, edge_dim=config.node_features)
                for _ in range(config.conv_layers)
            ]
        )
        
        self.edge_update_layers = nn.ModuleList( ## module not used
            [
                MatformerConv(in_channels=config.node_features, out_channels=config.node_features, heads=config.edge_layer_head, edge_dim=config.node_features)
                for _ in range(config.edge_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()
        self.Softplus = nn.Softplus()

        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(
                config.fc_features, 2
            )

        self.add_energies = nn.Sequential( #NOT USED
            nn.Linear(4, 4),
            nn.Softplus(),
            nn.Linear(4, 1)
        )

        self.rescale1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.rescale2 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.rescale3 = nn.Parameter(torch.zeros(1), requires_grad=True)


        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            if not self.zero_inflated:
                self.fc_out.bias.data = torch.tensor(
                    np.log(avg_gap), dtype=torch.float
                )
        elif config.link == "logit":
            self.link = torch.sigmoid

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
                # Get the reciprocal lattice indices of included k-vectors
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
                        use_pbc=self.use_pbc,
                        delta_k=self.delta_k,
                        k_rbf_values=self.k_rbf_values,
                    )
                    for i in range(config.conv_layers)
                ]
            )

        self.skip_connection_factor = (
            2.0 + float(self.use_ewald)
        ) ** (-0.5)    

    def forward(self, data) -> torch.Tensor:
        data, ldata, lattice = data
        # initial node features: atom feature network...
            
        node_features = self.atom_embedding(data.x)
        edge_feat = torch.norm(data.edge_attr, dim=1)

        # Number of atoms
        N = len(data.x)

        # Batch size
        num_batch = torch.max(data.batch).item() + 1

        # Index for each atom that specifies to which molecule in the batch it belongs.
        batch_seg = data.batch

        # Atomic positions
        tensor_list = [torch.tensor(array) for array in ldata.coords]  # Convert each array to a tensor
        R = torch.cat(tensor_list, dim=0)

        # Nuclear charges (atomic numbers) of atoms.
        atomic_numbers = ldata.atomic_number.squeeze()
        Z = atomic_numbers.to(torch.int)

        # Lattice information
        cell = torch.tensor(ldata.lattice_mat)
        
        edge_features = self.rbf(edge_feat)

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
   

        if self.use_ewald:
            dot = None  # These will be computed in first Ewald block and then passed
            sinc_damping = None  # on between later Ewald blocks (avoids redundant recomputation)
            for i in range(self.conv_layers):
                f_ewald, dot, sinc_damping = self.ewald_blocks[i](
                                node_features,
                                R.to(batch_seg.device),
                                k_grid,
                                num_batch,
                                batch_seg,
                                dot,
                                sinc_damping,
                            )
                node_features = self.skip_connection_factor * (self.att_layers[i](node_features, data.edge_index, edge_features) + f_ewald)
        else:
            for layer in self.att_layers:
                node_features = layer(node_features, data.edge_index, edge_features)

        # if self.angle_lattice: ## module not used
        #     lattice_len = torch.norm(lattice, dim=-1) # batch * 3 * 1
        #     lattice_edge = self.lattice_rbf(lattice_len.view(-1)).view(-1, 3 * 128) # batch * 3 * 128
        #     cos1 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,0,:] * lattice[:,1,:], dim=-1) / (torch.norm(lattice[:,0,:], dim=-1) * torch.norm(lattice[:,1,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
        #     cos2 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,0,:] * lattice[:,2,:], dim=-1) / (torch.norm(lattice[:,0,:], dim=-1) * torch.norm(lattice[:,2,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
        #     cos3 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,1,:] * lattice[:,2,:], dim=-1) / (torch.norm(lattice[:,1,:], dim=-1) * torch.norm(lattice[:,2,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
        #     lattice_emb = self.lattice_emb(torch.cat((lattice_edge, cos1, cos2, cos3), dim=-1))
        #     node_features = self.lattice_atom_emb(torch.cat((node_features, lattice_emb[data.batch]), dim=-1))
        
        # node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        # node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        # node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        # node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        # node_features = self.att_layers[4](node_features, data.edge_index, edge_features)

        # crystal-level readout
        #features = scatter(node_features, data.batch, dim=0, reduce="mean")

        # if self.angle_lattice:
        #     # features *= F.sigmoid(lattice_emb)
        #     features += lattice_emb

        # atom-level readout
        features = node_features    
        
        features = self.fc(features)

        out = self.fc_out(features)
        # if self.link:
        #     out = self.link(out)
        # if self.classification:
        #     out = self.softmax(out)

        # WE need output for both partial charges "qa" and atomic energy "ea"

        ea = out.narrow(-1, 0, 1).squeeze(-1)  # atomic energy
        qa = out.narrow(-1, 1, 1).squeeze(-1)  # partial charge

        # Index of atom i for all atomic pairs ij. Each pair must be specified as both ij and ji.
        idx_i = data.edge_index[0]
        # Same as idx_i, but for atom j.
        idx_j = data.edge_index[1]

        # Interatomic distances
        rij = edge_feat

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

        # Crystal-level readout
        if(self.learnable_parameters):
            ea = ea + self.rescale1 * ea_rep \
                    + self.rescale2 * ea_ele \
                    + self.rescale3 * ea_vdw

        else:
            ea = ea + ea_rep + ea_ele + ea_vdw      
        energy = scatter(ea, data.batch, dim=0, reduce="mean")
        return torch.squeeze(energy)