import json
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
from ase import Atoms
from ase.neighborlist import primitive_neighbor_list


def json_to_dict(json_file_path):
    """
    Method to load the json file to dictionary (provided a dictionary was converted to a json file )
    Arguments:
        json_file_path : str = path of json file
    Returns:
        loaded_dict : dict = dictionary associated with the json file
    """

    with open(json_file_path, "rb") as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict


def dict_to_json(dict_name, json_name):
    """
    Method to write dictionary to a json file
    Arguments:
        dict_name : str = path of dictionary to be written to json
        json_name : str = path of json file to be written
    Returns:
        none
    """
    with open(json_name, "w") as json_file:
        json.dump(dict_name, json_file)




def data_preparation(z_list, pos_list, wc_list1, ene_list, for_list, z_wannier: int = 8):
    """
    A method to prepare data for feeding to SchNet that stacks atomic positions and wannier centers across
    different configurations

    Arguments:
        z_list: np.ndarray = atomic_numbers of all atom indices in a given configuration
        pos_list: np.ndarray = position of each atoms across all configurations (configurations,number of atoms,3)
        z_wannier: atomic number of the atom around which wannier centers are to be created
    """

    atoms_list = []
    property_list = []
    for z_mol, positions, wanniers, energy, forces in zip(z_list, pos_list, wc_list1, ene_list, for_list):
        ats = Atoms(positions=positions, numbers=z_mol)

        properties = {
            "wan": wanniers,
            "wc_selector": np.array([1 if z == z_wannier else 0 for z in z_mol]),
            "energy": energy,
            "forces": forces,
        }
        property_list.append(properties)
        atoms_list.append(ats)

    return atoms_list, property_list


class Process_xyz_remsing:
    """
    Arguments:
        file_str: str = path of structure file (eg.position.xyz), POSCAR, CONTCAR
        file_init: str = path of the input control file with extension inp (for CP2K : "eg. init.xyz")
        file_force: str = path of the input control file with extension inp (force  : "W64-bulk-W64-forces-1_0.xyz")
        self.phrase: str = phrase that we want to search in the file
        num_wan: int = number of wannier centers per formula unit. eg. for H20, there are 4 WC around
                        oxygen atom (most electronegative), default = 4
        pbe: bool = True/False, True indicates consideration of periodic boundary condition : default: True
        format_in: str = input file type of first argument (file_str): default = 'xyz'
    """

    def __init__(self, file_str,file_init, file_force,num_wan=4, pbc=True,phrase='TotEnergy', format_in="xyz"):
        self.file_str = file_str
        self.file_init = file_init
        self.file_force = file_force
        self.num_wan = num_wan
        self.phrase = phrase
        self.pbc = pbc
        self.format_in = format_in
        # print(lattice_vectors)

    def get_line(self):
        """
        searches for a line containing the phrase in the file and returns first line containing it.
        """
        with open(self.file_init) as f:
            for i, line in enumerate(f):
                if self.phrase in line:
                    return line

    def get_energy_cell_lattice(self):
        """
        writes the total energy (eV) and cell vectors (A)  
        """    
        a1=self.get_line()
        a2=list(filter(lambda x: x.strip(), a1.split("=")))
        hartree_to_ev=27.211396
        bohr_to_angs=0.529177208
        energy_hartree=float(a2[1].split(" ")[0])
        lat=a2[5].split('"')[1]
        cell_bohr = np.array([float(x) for x in lat.split()]).reshape(3,3)
        energy_ev=energy_hartree*hartree_to_ev
        cell_angs=cell_bohr*bohr_to_angs
        return energy_ev,cell_angs

    def get_forces(self):
        """
        writes total forces (eV/A)
        """
        max_rows=int(next(open(self.file_init, 'r')))
        au_to_ev_per_ang=51.42208619083232
        forces_array=np.loadtxt(self.file_force,usecols=(3,4,5),skiprows=4,max_rows=max_rows)
        return forces_array*au_to_ev_per_ang

    def get_neigh(
        self,
        coords: np.ndarray,
        r_cut: float,
        pbc: Union[bool, Tuple[bool, bool, bool]] = True,
        cell: np.ndarray = None,
        self_interaction: bool = False,
        periodic_self_interaction: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create neighbor list for all points in a point cloud.

        Args:
            coords: (N, 3) array of positions, where N is the number of points.
            r_cut: cutoff distance for neighbor finding.
            pbc: Whether to use periodic boundary conditions. If a list of bools, then
                each entry corresponds to a supercell vector. If a single bool, then the
                same value is used for all supercell vectors.
            cell: (3, 3) array of supercell vectors. cell[i] is the i-th supercell vector.
                Ignored if `pbc == False` or pbc == None`.
            self_interaction: Whether to include self-interaction, i.e. an atom being the
                neighbor of itself in the neighbor list. Should be False for most
                applications.
            periodic_self_interaction: Whether to include interactions of an atom with its
                periodic images. Should be True for most applications.

        Returns:
            edge_index: (2, num_edges) array of edge indices. The first row contains the
                i atoms (center), and the second row contains the j atoms (neighbor).
            shift_vec: (num_edges, 3) array of shift vectors. The number of cell boundaries
                crossed by the bond between atom i and j. The distance vector between atom
                j and atom i is given by `coords[j] - coords[i] + shift_vec.dot(cell)`.
            num_neigh: (N,) array of the number of neighbors for each atom.
        """
        if isinstance(pbc, bool):
            pbc = [pbc] * 3

        if not np.any(pbc):
            self_interaction = False
            periodic_self_interaction = False

        if cell is None:
            if not np.any(pbc):
                cell = np.eye(3)  # dummy cell to use
            else:
                raise RuntimeError("`cell` vectors not provided")

        (
            first_idx,
            second_idx,
            abs_distance,
            distance_vector,
            shift_vec,
        ) = primitive_neighbor_list(
            "ijdDS",
            pbc=pbc,
            cell=cell,
            positions=coords,
            cutoff=r_cut,
            self_interaction=periodic_self_interaction,
        )

        # remove self interactions
        if periodic_self_interaction and (not self_interaction):
            bad_edge = first_idx == second_idx
            bad_edge &= np.all(shift_vec == 0, axis=1)
            keep_edge = ~bad_edge
            if not np.any(keep_edge):
                raise RuntimeError(
                    "After removing self interactions, no edges remain in this system."
                )
            first_idx = first_idx[keep_edge]
            second_idx = second_idx[keep_edge]
            abs_distance = abs_distance[keep_edge]
            distance_vector = distance_vector[keep_edge]
            shift_vec = shift_vec[keep_edge]

        # number of neighbors for each atom
        num_neigh = np.bincount(first_idx)

        # Some atoms with large index may not have neighbors due to the use of bincount.
        # As a concrete example, suppose we have 5 atoms and first_idx is [0,1,1,3,3,3,3],
        # then bincount will be [1, 2, 0, 4], which means atoms 0,1,2,3 have 1,2,0,4
        # neighbors respectively. Although atom 2 is handled by bincount, atom 4 is not.
        # The below part is to make this work.
        if len(num_neigh) != len(coords):
            extra = np.zeros(len(coords) - len(num_neigh), dtype=int)
            num_neigh = np.concatenate((num_neigh, extra))

        edge_index = np.vstack((first_idx, second_idx))

        return edge_index, num_neigh, abs_distance, distance_vector, shift_vec

    def get_structure(self):
        import ase.io as asi

        energy,cell = self.get_energy_cell_lattice()
        struct_obj = asi.read(filename=self.file_str, format=self.format_in)
        struct_obj.set_pbc(self.pbc)
        struct_obj.set_cell(cell)
        return struct_obj, cell

    def atom_number_positions(self):
        struct_obj, cell = self.get_structure()
        z_mol = struct_obj.numbers[np.array(np.where(struct_obj.numbers != 0))][0]
        pos_mol = struct_obj.get_positions()[
            np.array(np.where(struct_obj.numbers != 0))
        ][0]
        return z_mol, pos_mol

    def wannier_centers(self):
        import numpy as np

        struct_obj, cell = self.get_structure()

        # The following blocks are mode relevant to water system only. This part needs modification for generalization
        oxy_positions = struct_obj.get_positions()[np.where(struct_obj.numbers == 8)]
        wan_positions = struct_obj.get_positions()[np.where(struct_obj.numbers == 0)]
        coords1 = np.concatenate((oxy_positions, wan_positions), axis=0)
        (
            edge_index,
            num_neigh,
            abs_distance,
            distance_vector,
            shift_vec,
        ) = self.get_neigh(
            coords=coords1,
            r_cut=0.74,
            cell=cell,
            pbc=True,
            self_interaction=False,
            periodic_self_interaction=True,
        )
        wc_neigh = num_neigh[: len(oxy_positions)]
        lst_wan = []
        sum1 = 0
        i: int
        for i, entries in enumerate(num_neigh[: len(oxy_positions)]):
            # Uncomment the following line to define absolute position of wannier center
            # lst1 = (np.sum(distance_vector[sum1 : sum1 + entries], axis=0) / entries).reshape(1, 3) + oxy_positions[i : i + 1,]
            # The following line will define position of wannier center relative to oxygen atom.
            lst1 = (
                np.sum(distance_vector[sum1 : sum1 + entries], axis=0) / entries
            ).reshape(1, 3)
            lst_wan.append(lst1[0])
            sum1 += entries

        lst_wan = np.array(lst_wan)

        return lst_wan, oxy_positions, wc_neigh

    def write_xyz(self, file_out="outfile_ret.xyz", format_out="xyz"):
        import ase.io as asi
        from ase import Atoms

        lst_wan, oxy_positions, wc_neigh = self.wannier_centers()
        # print(z_mol.shape)
        # print(len(pos_oxygen),len(lst_wan))
        new_str = Atoms(
            "O" + str(len(oxy_positions)) + "X" + str(len(lst_wan)),
            np.concatenate((oxy_positions, lst_wan), axis=0),
            pbc=True,
            cell=self.get_lattice_vectors(),
        )
        asi.write(file_out, images=new_str, format=format_out)



def read_data(path):
    """

    Args:
        path: Path to the directory containing the data

    Returns:
    """
    na_list = []
    neigh_mismath_list = []
    dict_out = {}
    dict_out["coords"] = {}
    keys_list = [
        "coords",
        "atomic_number",
        "cell",
        "pbc",
        "energy",
        "forces",
        "wannier_center",
        "atomic_selector",
    ]
    for entries in keys_list:
        dict_out[str(entries)] = {}
    print(dict_out.keys())
    lst_dir = [x for x in os.listdir(path) if "." not in x]
    for dir in lst_dir:
        file_path1 = path + str(dir)
        if "W64-bulk-HOMO_centers_s1-1_0.xyz" in os.listdir(file_path1):
            file_str = file_path1 + "/W64-bulk-HOMO_centers_s1-1_0.xyz"
            file_init = file_path1 + "/init.xyz"
            file_force = file_path1 +"/W64-bulk-W64-forces-1_0.xyz"
            pxr = Process_xyz_remsing(file_str = file_str, file_init = file_init, file_force = file_force)
            pbc = pxr.pbc
            lst_wan, oxy_positions, wc_neigh = pxr.wannier_centers()
            if list(np.unique(wc_neigh)) == [4]:
                z_mol, pos_mol = pxr.atom_number_positions()
                energy, cell = pxr.get_energy_cell_lattice()
                forces=pxr.get_forces()
                oxygen_mask = z_mol == 8
                # print("mask_shape", oxygen_mask.shape)
                result = np.zeros_like(pos_mol)
                # print("lst_wan_shape", lst_wan.shape)
                result[oxygen_mask] = lst_wan[z_mol[oxygen_mask] == 8]
                # print(z_mol[:6],result[:6,:],lst_wan[:2,:])

                val_list = [
                    pos_mol,
                    z_mol,
                    cell,
                    np.array([pbc, pbc, pbc]),
                    energy,
                    forces,
                    result,
                    oxygen_mask,
                ]

                for i in range(len(keys_list)):
                    dict_out[keys_list[i]][str(dir)] = val_list[i]
            else:
                neigh_mismath_list.append(dir)
        else:
            na_list.append(dir)
    print("Number of na:", len(na_list))
    print("Number of neighbor mismatch for given cutoff:", len(neigh_mismath_list))

    return dict_out


if __name__ == "__main__":
    # path = "/Users/sadhik22/Downloads/train_test_configs_orig/D0/"
    path = "/project/wen/sadhik22/schnet_training/wannier_centers/dataset/train_test_configs_orig/D0/"
    dict_out = read_data(path)

    print("number of data points", len(dict_out["cell"]))

    df = pd.DataFrame(dict_out)
    df.to_json("wannier_updated.json")
