import torch
import numpy as np


def assign_charges_to_grid(positions, charges, grid_size, cell_vectors):
    grid = torch.zeros((grid_size, grid_size, grid_size), dtype=torch.complex128)
    N = len(charges)
    inv_cell_vectors = torch.inverse(cell_vectors)

    for i in range(N):
        fractional_position = torch.matmul(positions[i], inv_cell_vectors)
        fractional_position -= torch.floor(fractional_position)
        grid_pos = fractional_position * grid_size
        grid_indices = torch.floor(grid_pos).long()

        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    shift = torch.tensor([dx, dy, dz], dtype=torch.float64)
                    weight = torch.prod(1.0 - torch.abs(shift - grid_pos % 1))
                    index = (grid_indices + shift.long()) % grid_size
                    grid[index[0], index[1], index[2]] += weight * charges[i]

    return grid


def solve_poisson(grid, alpha):
    grid_size = grid.shape[0]
    k_vectors = torch.fft.fftfreq(grid_size, d=1.0 / grid_size)
    kx, ky, kz = torch.meshgrid(k_vectors, k_vectors, k_vectors, indexing='ij')
    k_squared = kx ** 2 + ky ** 2 + kz ** 2
    k_squared[0, 0, 0] = 1  # Avoid division by zero
    potential_grid = torch.fft.ifftn(torch.fft.fftn(grid) / k_squared) * torch.exp(-k_squared / (4 * alpha ** 2))
    return potential_grid


def interpolate_forces_from_grid(positions, potential_grid, grid_size, cell_vectors):
    forces = torch.zeros_like(positions, dtype=torch.float64)
    N = positions.shape[0]
    inv_cell_vectors = torch.inverse(cell_vectors)

    # Use only the real part of the potential grid
    potential_grid_real = torch.real(potential_grid)

    for i in range(N):
        fractional_position = torch.matmul(positions[i], inv_cell_vectors)
        fractional_position -= torch.floor(fractional_position)
        grid_pos = fractional_position * grid_size
        grid_indices = torch.floor(grid_pos).long()

        potential_grad = torch.zeros(3, dtype=torch.float64)
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    shift = torch.tensor([dx, dy, dz], dtype=torch.float64)
                    weight = torch.prod(1.0 - torch.abs(shift - grid_pos % 1))
                    index = (grid_indices + shift.long()) % grid_size
                    index = tuple(index.tolist())  # Convert to tuple for indexing

                    # Manually compute the finite differences
                    for dim in range(3):
                        delta = torch.zeros(3, dtype=torch.float64)
                        delta[dim] = 1.0
                        plus_index = tuple(((grid_indices + shift.long() + delta.long()) % grid_size).tolist())
                        minus_index = tuple(((grid_indices + shift.long() - delta.long()) % grid_size).tolist())
                        grad = (potential_grid_real[plus_index] - potential_grid_real[minus_index]) / 2.0
                        potential_grad[dim] += weight * grad

        forces[i] = -potential_grad

    return forces


def compute_reciprocal_energy(grid, potential_grid, cell_volume):
    # Compute reciprocal space energy
    reciprocal_energy = 0.5 * torch.sum(grid * torch.conj(potential_grid)).real / cell_volume
    return reciprocal_energy


def compute_direct_sum(cell_vectors, positions, charges, alpha):
    N = len(charges)
    energy_direct = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rij = positions[i] - positions[j]
            rij -= torch.round(rij @ torch.inverse(cell_vectors)) @ cell_vectors
            r = torch.norm(rij)
            if r > 0:
                energy_direct += charges[i] * charges[j] * torch.erfc(alpha * r) / r
    return energy_direct


def compute_self_energy(charges, alpha):
    return -alpha / torch.sqrt(torch.tensor(np.pi, dtype=torch.float64)) * torch.sum(charges ** 2)


def pppm_method(cell_vectors, positions, charges, alpha=0.1, grid_size=32):
    cell_volume = torch.det(cell_vectors)

    # Direct sum energy
    energy_direct = compute_direct_sum(cell_vectors, positions, charges, alpha)

    # Assign charges to grid and solve Poisson's equation
    grid = assign_charges_to_grid(positions, charges, grid_size, cell_vectors)
    potential_grid = solve_poisson(grid, alpha)

    # Compute reciprocal space energy
    reciprocal_energy = compute_reciprocal_energy(grid, potential_grid, cell_volume)

    # Self energy
    energy_self = compute_self_energy(charges, alpha)

    # Total energy
    ewald_energy = energy_direct + reciprocal_energy + energy_self

    # Compute forces
    forces = interpolate_forces_from_grid(positions, potential_grid, grid_size, cell_vectors)

    return ewald_energy, reciprocal_energy, energy_direct, energy_self, forces
