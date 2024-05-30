from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
import pymatgen.analysis.ewald as ewl

import numpy as np


def ewald_sum(atomic_numbers,positions,cell_vectors,dict_oxidation):
        lattice = Lattice(cell_vectors)
        struct = Structure(lattice, atomic_numbers, positions)
        struct.add_oxidation_state_by_element(dict_oxidation)
        sum_ewl=ewl.EwaldSummation(structure=struct,compute_forces=False)
        return sum_ewl

# Define lattice parameters (dimensions) of the box
box_dimensions = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]  # Example dimensions, adjust as needed

# List of atoms and their positions
#atoms = ["Na", "Cl"]  # Example atomic species

atomic_numbers = [1, 1, 8 , 110]
positions = [[0, 0, 0], [2.5, 2.5, 2.5], [5, 5, 5], [6, 6, 6]]  # Example positions, adjust as needed

# Dictionary for updating the oxidation states. Oxidation states based on the number of valance electrons in pp
dict_oxidation={'H':1,'O':6,'Ds':-8}
ewl_sum=ewald_sum(np.array(atomic_numbers),np.array(positions),np.array(box_dimensions),dict_oxidation)
print(ewl_sum.total_energy)
print(ewl_sum.reciprocal_space_energy)
