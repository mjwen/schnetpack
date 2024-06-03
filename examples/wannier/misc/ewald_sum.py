import torch
import numpy as np

class Ewald_PPPM:
    def __init__(self, cell_vectors, positions, charges, alpha=0.1, grid_size=32):
        """
        Initialize the Ewald_PPPM class.

        Args:
            cell_vectors (torch.Tensor): Lattice vectors of the cell.
            positions (torch.Tensor): Positions of atoms within the cell.
            charges (torch.Tensor): Charges on each atom.
            alpha (float, optional): Ewald parameter. Default is 0.1.
            grid_size (int, optional): Size of the grid for PPPM method. Default is 32.
        """
        self.cell_vectors = cell_vectors
        self.positions = positions
        self.charges = charges
        self.alpha = alpha
        self.grid_size = grid_size
        self.cell_volume = torch.det(cell_vectors)
        self.inv_cell_vectors = torch.inverse(cell_vectors)

    def assign_charges_to_grid(self):
        """
        Assign charges to the grid.

        Returns:
            torch.Tensor: Grid with assigned charges.
        """
        grid = torch.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=torch.complex128)
        N = len(self.charges)

        for i in range(N):
            fractional_position = torch.matmul(self.positions[i], self.inv_cell_vectors)
            fractional_position -= torch.floor(fractional_position)
            grid_pos = fractional_position * self.grid_size
            grid_indices = torch.floor(grid_pos).long()

            for dx in range(2):
                for dy in range(2):
                    for dz in range(2):
                        shift = torch.tensor([dx, dy, dz], dtype=torch.float64)
                        weight = torch.prod(1.0 - torch.abs(shift - grid_pos % 1))
                        index = (grid_indices + shift.long()) % self.grid_size
                        grid[index[0], index[1], index[2]] += weight * self.charges[i]

        return grid

    def solve_poisson(self, grid):
        """
        Solve Poisson's equation on the grid.

        Args:
            grid (torch.Tensor): Grid with assigned charges.

        Returns:
            torch.Tensor: Grid with computed potential.
        """
        k_vectors = torch.fft.fftfreq(self.grid_size, d=1.0 / self.grid_size)
        kx, ky, kz = torch.meshgrid(k_vectors, k_vectors, k_vectors, indexing='ij')
        k_squared = kx ** 2 + ky ** 2 + kz ** 2
        k_squared[0, 0, 0] = 1  # Avoid division by zero
        potential_grid = torch.fft.ifftn(torch.fft.fftn(grid) / k_squared) * torch.exp(-k_squared / (4 * self.alpha ** 2))
        return potential_grid

    def interpolate_forces_from_grid(self, potential_grid):
        """
        Interpolate forces from the potential grid.

        Args:
            potential_grid (torch.Tensor): Grid with computed potential.

        Returns:
            torch.Tensor: Interpolated forces on the atoms.
        """
        forces = torch.zeros_like(self.positions, dtype=torch.float64)
        N = self.positions.shape[0]

        # Use only the real part of the potential grid
        potential_grid_real = torch.real(potential_grid)

        for i in range(N):
            fractional_position = torch.matmul(self.positions[i], self.inv_cell_vectors)
            fractional_position -= torch.floor(fractional_position)
            grid_pos = fractional_position * self.grid_size
            grid_indices = torch.floor(grid_pos).long()

            potential_grad = torch.zeros(3, dtype=torch.float64)
            for dx in range(2):
                for dy in range(2):
                    for dz in range(2):
                        shift = torch.tensor([dx, dy, dz], dtype=torch.float64)
                        weight = torch.prod(1.0 - torch.abs(shift - grid_pos % 1))
                        index = (grid_indices + shift.long()) % self.grid_size
                        index = tuple(index.tolist())  # Convert to tuple for indexing

                        # Manually compute the finite differences
                        for dim in range(3):
                            delta = torch.zeros(3, dtype=torch.float64)
                            delta[dim] = 1.0
                            plus_index = tuple(((grid_indices + shift.long() + delta.long()) % self.grid_size).tolist())
                            minus_index = tuple(((grid_indices + shift.long() - delta.long()) % self.grid_size).tolist())
                            grad = (potential_grid_real[plus_index] - potential_grid_real[minus_index]) / 2.0
                            potential_grad[dim] += weight * grad

            forces[i] = -potential_grad

        return forces

    def compute_reciprocal_energy(self, grid, potential_grid):
        """
        Compute the reciprocal space energy.

        Args:
            grid (torch.Tensor): Grid with assigned charges.
            potential_grid (torch.Tensor): Grid with computed potential.

        Returns:
            float: Reciprocal space energy.
        """
        reciprocal_energy = 0.5 * torch.sum(grid * torch.conj(potential_grid)).real / self.cell_volume
        return reciprocal_energy

    def compute_direct_sum(self):
        """
        Compute the direct sum energy.

        Returns:
            float: Direct sum energy.
        """
        N = len(self.charges)
        energy_direct = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                rij = self.positions[i] - self.positions[j]
                rij -= torch.round(rij @ self.inv_cell_vectors) @ self.cell_vectors
                r = torch.norm(rij)
                if r > 0:
                    energy_direct += self.charges[i] * self.charges[j] * torch.erfc(self.alpha * r) / r
        return energy_direct

    def compute_self_energy(self):
        """
        Compute the self-energy.

        Returns:
            float: Self-energy.
        """
        return -self.alpha / torch.sqrt(torch.tensor(np.pi, dtype=torch.float64)) * torch.sum(self.charges ** 2)

    def pppm_method(self):
        """
        Compute the Ewald sum using the PPPM method.

        Returns:
            tuple: Total Ewald energy, reciprocal energy, direct energy, self energy, and forces.
        """
        # Direct sum energy
        energy_direct = self.compute_direct_sum()

        # Assign charges to grid and solve Poisson's equation
        grid = self.assign_charges_to_grid()
        potential_grid = self.solve_poisson(grid)

        # Compute reciprocal space energy
        reciprocal_energy = self.compute_reciprocal_energy(grid, potential_grid)

        # Self energy
        energy_self = self.compute_self_energy()

        # Total energy
        ewald_energy = energy_direct + reciprocal_energy + energy_self

        # Compute forces
        forces = self.interpolate_forces_from_grid(potential_grid)

        return ewald_energy, reciprocal_energy, energy_direct, energy_self, forces


class Ewald_Direct:
    def __init__(self, cell_vectors, positions, charges, kmax=12, alpha=0.0629):
        """
        Initialize the Ewald_Direct class.

        Args:
            cell_vectors (torch.Tensor): Lattice vectors of the cell.
            positions (torch.Tensor): Positions of atoms within the cell.
            charges (torch.Tensor): Charges on each atom.
            kmax (int, optional): Maximum k-vector for reciprocal sum. Default is 12.
            alpha (float, optional): Ewald parameter. Default is 0.0629.
        """
        self.cell_vectors = cell_vectors
        self.positions = positions
        self.charges = charges
        self.kmax = kmax
        self.alpha = alpha
        self.inv_cell_vectors = torch.inverse(cell_vectors)
        self.volume = torch.det(cell_vectors)

    def direct_sum(self):
        """
        Compute the direct sum energy.

        Returns:
            float: Direct sum energy.
        """
        N = len(self.charges)
        energy_direct = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                rij = self.positions[j] - self.positions[i]
                rij -= torch.round(rij @ self.inv_cell_vectors) @ self.cell_vectors
                r = torch.norm(rij)
                energy_direct += self.charges[i] * self.charges[j] / r
        return energy_direct

    def reciprocal_sum(self):
        """
        Compute the reciprocal space energy.

        Returns:
            float: Reciprocal space energy.
        """
        energy_reciprocal = 0.0
        for h in range(-self.kmax, self.kmax + 1):
            for k in range(-self.kmax, self.kmax + 1):
                for l in range(-self.kmax, self.kmax + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    kvec = 2 * torch.pi * torch.tensor([h, k, l], dtype=self.cell_vectors.dtype) @ self.inv_cell_vectors
                    ksq = torch.dot(kvec, kvec)
                    if ksq > 0:
                        exp_term = torch.exp(1j * torch.matmul(self.positions, kvec))
                        structure_factor = torch.sum(self.charges * exp_term)
                        energy_reciprocal += (4 * torch.pi / (self.volume * ksq)) * torch.abs(structure_factor) ** 2 * torch.exp(-ksq / (4 * self.alpha ** 2))
        return energy_reciprocal

    def self_energy(self):
        """
        Compute the self-energy.

        Returns:
            float: Self-energy.
        """
        energy_self = -torch.pi * torch.sum(self.charges ** 2) / (self.alpha * torch.sqrt(self.volume))
        return energy_self

    def compute_ewald_sum(self):
        """
        Compute the Ewald sum using the direct method.

        Returns:
            tuple: Total Ewald energy, direct sum energy, reciprocal sum energy, and self energy.
        """
        energy_direct = self.direct_sum()
        energy_reciprocal = self.reciprocal_sum()
        energy_self = self.self_energy()

        ewald_sum = energy_direct + energy_reciprocal + energy_self
        return ewald_sum, energy_direct, energy_reciprocal, energy_self
