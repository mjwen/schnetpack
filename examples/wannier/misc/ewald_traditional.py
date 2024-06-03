
import torch


# Torch version for comparison
def compute_ewald_sum(cell_vectors, positions, charges, kmax=12, alpha=0.0629):
    def direct_sum(cell_vectors, positions, charges):
        N = len(charges)
        energy_direct = 0.0
        for i in range(N):
            for j in range(i+1, N):
                rij = positions[j] - positions[i]
                rij -= torch.round(rij @ torch.inverse(cell_vectors)) @ cell_vectors
                r = torch.norm(rij)
                energy_direct += charges[i] * charges[j] / r
        return energy_direct

    def reciprocal_sum(cell_vectors, positions, charges, kmax, alpha):
        energy_reciprocal = 0.0
        volume = torch.det(cell_vectors)
        for h in range(-kmax, kmax+1):
            for k in range(-kmax, kmax+1):
                for l in range(-kmax, kmax+1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    kvec = 2 * torch.pi * torch.tensor([h, k, l], dtype=cell_vectors.dtype) @ torch.inverse(cell_vectors)
                    ksq = torch.dot(kvec, kvec)
                    if ksq > 0:
                        exp_term = torch.exp(1j * torch.matmul(positions, kvec))
                        structure_factor = torch.sum(charges * exp_term)
                        energy_reciprocal += (4 * torch.pi / (volume * ksq)) * torch.abs(structure_factor)**2 * torch.exp(-ksq / (4 * alpha**2))
        return energy_reciprocal

    def self_energy(cell_vectors, charges, alpha):
        volume = torch.det(cell_vectors)
        energy_self = -torch.pi * torch.sum(charges**2) / (alpha * torch.sqrt(volume))
        return energy_self

    energy_direct = direct_sum(cell_vectors, positions, charges)
    energy_reciprocal = reciprocal_sum(cell_vectors, positions, charges, kmax, alpha)
    energy_self = self_energy(cell_vectors, charges, alpha)

    ewald_sum = energy_direct + energy_reciprocal + energy_self

    return ewald_sum, energy_direct, energy_reciprocal, energy_self
