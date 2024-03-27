import torch
import numpy as np
from ewald_auxiliary.radial_basis import RadialBasis

def get_k_index_product_set(num_k_x, num_k_y, num_k_z):
    # Get a box of k-lattice indices around (0,0,0)
    k_index_sets = (
        torch.arange(-num_k_x, num_k_x + 1, dtype=torch.float),
        torch.arange(-num_k_y, num_k_y + 1, dtype=torch.float),
        torch.arange(-num_k_z, num_k_z + 1, dtype=torch.float),
    )
    k_index_product_set = torch.cartesian_prod(*k_index_sets)
    # Cut the box in half (we will always assume point symmetry)
    k_index_product_set = k_index_product_set[
        k_index_product_set.shape[0] // 2 + 1 :
    ]

    # Amount of k-points
    num_k_degrees_of_freedom = k_index_product_set.shape[0]

    return k_index_product_set, num_k_degrees_of_freedom


def pos_svd_frame(data):
    pos = data.pos
    batch = data.batch
    batch_size = int(batch.max()) + 1

    with torch.cuda.amp.autocast(False):
        rotated_pos_list = []
        for i in range(batch_size):
            # Center each structure around mean position
            pos_batch = pos[batch == i]
            pos_batch = pos_batch - pos_batch.mean(0)

            # Rotate each structure into its SVD frame
            # (only can do this if structure has at least 3 atoms,
            # i.e., the position matrix has full rank)
            if pos_batch.shape[0] > 2:
                U, S, V = torch.svd(pos_batch)
                rotated_pos_batch = torch.matmul(pos_batch, V)

            else:
                rotated_pos_batch = pos_batch

            rotated_pos_list.append(rotated_pos_batch)

        pos = torch.cat(rotated_pos_list)

    return pos


def x_to_k_cell(cells):

    cross_a2a3 = torch.cross(cells[:, 1], cells[:, 2], dim=-1)
    cross_a3a1 = torch.cross(cells[:, 2], cells[:, 0], dim=-1)
    cross_a1a2 = torch.cross(cells[:, 0], cells[:, 1], dim=-1)
    vol = torch.sum(cells[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    b1 = 2 * np.pi * cross_a2a3 / vol
    b2 = 2 * np.pi * cross_a3a1 / vol
    b3 = 2 * np.pi * cross_a1a2 / vol

    bcells = torch.stack((b1, b2, b3), dim=1)

    return (bcells, vol[:, 0])

def get_k_voxel_grid(k_cutoff, delta_k, num_k_rbf):

    # Get indices for a cube of k-lattice sites containing the cutoff sphere
    num_k = k_cutoff / delta_k
    k_index_product_set, _ = get_k_index_product_set(num_k, num_k, num_k)

    # Orthogonal k-space basis, norm delta_k
    k_cell = torch.tensor(
        [[delta_k, 0, 0], [0, delta_k, 0], [0, 0, delta_k]], dtype=torch.float
    )

    # Translate lattice indices into k-vectors
    k_grid = torch.matmul(k_index_product_set, k_cell)

    # Prune all k-vectors outside the cutoff sphere
    k_grid = k_grid[torch.sum(k_grid**2, dim=-1) <= k_cutoff**2]

    # Probably quite arbitrary, for backwards compatibility with scaling
    # yaml files produced with old Ewald Message Passing code
    k_offset = 0.1 if num_k_rbf <= 48 else 0.25

    # Evaluate a basis of Gaussian RBF on the k-vectors
    k_rbf_values = RadialBasis(
        num_radial=num_k_rbf,
        # Avoids zero or extremely small RBF values (there are k-points until
        # right at the cutoff, where all RBF would otherwise drop to 0)
        cutoff=k_cutoff + k_offset,
        rbf={"name": "gaussian"},
        envelope={"name": "polynomial", "exponent": 5},
    )(
        torch.linalg.norm(k_grid, dim=-1)
    )  # Tensor of shape (N_k, N_RBF)

    num_k_degrees_of_freedom = k_rbf_values.shape[-1]

    return k_grid, k_rbf_values, num_k_degrees_of_freedom