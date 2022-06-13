#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASE Calculator to combine QM and MM forces and energies.
"""
import sys

import numpy as np
import ase
import ase.calculators.calculator

from .utils import *


class QMMMHamiltonian(ase.calculators.calculator.Calculator):
    """ 
    ASE Calculator.

    Modeled after SchNetPack calculator.

    Parameters
    ----------
    openmm_interface: OpenMMInterface object
        OpenMMInterface object containing all the needed info for
        getting forces from OpenMM.
    psi4_interface: Psi4Interface object
        Psi4Interface object containing all the needed info for getting
        forces from Psi4.
    qm_atoms_list: list of int
        List containing the integer indices of the QM atoms
    embedding_cutoff: float
        Cutoff distance, in Angstroms, within which molecules will be
        electrostatically embedded in the QM calculation.
    residue_atom_lists: list of list of int
        Residue list containing lists of atom indices in each residue
    **kwargs: dict
        Additional args for ASE base calculator
    """
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(self, openmm_interface, psi4_interface, qm_atoms_list,
                 embedding_cutoff, residue_atom_lists, **kwargs):
        ase.calculators.calculator.Calculator.__init__(self, **kwargs)
        self.openmm_interface = openmm_interface
        self.psi4_interface = psi4_interface
        self.qm_atoms_list = qm_atoms_list
        self.embedding_cutoff = embedding_cutoff
        self.residue_atom_lists = residue_atom_lists
        #self.has_periodic_box = self.openmm_interface.has_periodic_box
        self.has_periodic_box = True
        self.energy_units = ase.units.kJ / ase.units.mol
        self.forces_units = ase.units.kJ / ase.units.mol / ase.units.Angstrom
        self.frame = 0

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        """
        Obtains the total energy and forces using the above interfaces.

        Parameters
        ------------
        atoms: ASE Atoms object, Optional, default=None
            Atoms object containing coordinates.
        properties: list of str, Optional, default=['energy']
            Not used.
        system_changes: list, Optional, 
                default=ase.calculators.calculator.all_changes
            List of changes for ASE.
        """
        result = {}
        # Shift atoms to ensure that molecules are not broken up.
        if self.has_periodic_box:
            atoms.wrap()
            shift_array = self.make_molecule_whole(
                atoms.get_positions(), atoms.get_cell())
            atoms.positions -= shift_array
        # Set gemoetry for the QM/MM Psi4 calculation.
        self.embed_electrostatics(atoms.positions, atoms.get_cell())
        self.psi4_interface.generate_geometry(self.embedding_list, 
                                              self.delr_vector_list,
                                              atoms.positions)
        ase.calculators.calculator.Calculator.calculate(self, atoms)
        self.openmm_interface.set_positions(atoms.get_positions())
        self.openmm_interface.embedding_list = self.embedding_list
        self.openmm_interface.delr_vector_list = self.delr_vector_list
        with open(self.logger, "a") as fh:
            fh.write("\n" + "-"*29 + "Frame " + "0"*(8-len(str(self.frame))) + str(self.frame) + "-"*29 + "\n")
            category = "Kinetic Energy"
            value = str(atoms.get_kinetic_energy()*96.4869)
            left, right = value.split(".")
            fh.write(category + ":" + " "*(31-len(left)-len(category)) + left + "." + right[0] + " kJ/mol\n")
        openmm_energy, openmm_forces = self.openmm_interface.compute_energy()
        psi4_energy, psi4_forces = self.psi4_interface.compute_energy()
        qm_forces = psi4_forces[0:len(self.qm_atoms_list),:]
        em_forces = psi4_forces[len(self.qm_atoms_list):,:]
        # Add Psi4 electrostatic forces and energy onto OpenMM forces
        # and energy for QM atoms.
        for i, qm_force in zip(self.qm_atoms_list, qm_forces):
            for j in range(3):
                openmm_forces[i,j] += qm_force[j]
        # Remove double-counting from embedding forces and energy.
        j = 0
        qm_centroid = [sum([atoms.positions[i][j] for i in self.qm_atoms_list])
                       / len(self.qm_atoms_list) for j in range(3)]
        dc_energy = 0.0
        for residue, offset in zip(self.embedding_list, self.delr_vector_list):
            for atom in residue:
                co_forces = [0,0,0]
                for i in self.qm_atoms_list:
                    x = (atoms.positions[atom][0] + offset[0] - (atoms.positions[i][0] - qm_centroid[0])) * 1.88973
                    y = (atoms.positions[atom][1] + offset[1] - (atoms.positions[i][1] - qm_centroid[1])) * 1.88973
                    z = (atoms.positions[atom][2] + offset[2] - (atoms.positions[i][2] - qm_centroid[2])) * 1.88973
                    dr = (x**2 + y**2 + z**2)**0.5
                    q_prod = atoms.charges[i] * atoms.charges[atom]
                    co_forces[0] += 1.88973 * 2625.5 * x * q_prod * dr**-3
                    co_forces[1] += 1.88973 * 2625.5 * y * q_prod * dr**-3
                    co_forces[2] += 1.88973 * 2625.5 * z * q_prod * dr**-3
                    dc_energy += 2625.5 * q_prod * dr**-1
                for i in range(3):
                    openmm_forces[atom,i] += em_forces[j][i]
                    openmm_forces[atom,i] -= co_forces[i]
                j += 1
        with open(self.logger, "a") as fh:
            category = "Psi4 Energy"
            value = str(psi4_energy)
            left, right = value.split(".")
            fh.write(category + ":" + " "*(31-len(left)-len(category)) + left + "." + right[0] + " kJ/mol\n")
            category = "Correction Energy"
            value = str(-dc_energy)
            left, right = value.split(".")
            fh.write(category + ":" + " "*(31-len(left)-len(category)) + left + "." + right[0] + " kJ/mol\n")
        openmm_energy += psi4_energy - dc_energy
        with open(self.logger, "a") as fh:
            category = "Total Energy"
            value = str(atoms.get_kinetic_energy()*96.4869 + openmm_energy)
            left, right = value.split(".")
            fh.write(category + ":" + " "*(31-len(left)-len(category)) + left + "." + right[0] + " kJ/mol\n")
        self.frame += 1
        result["energy"] = openmm_energy * self.energy_units
        result["forces"] = openmm_forces * self.forces_units
        self.results = result

    def make_molecule_whole(self, position_array, box):
        """
        Atoms are wrapped to stay inside of the periodic box. 

        This function ensures molecules are not broken up by a periodic
        boundary, as OpenMM electrostatics will be incorrect if all
        atoms in a molecule are not on the same side of the periodic
        box.  Assumes isotropic box.

        Parameters
        ----------
        position_array: NumPy array
            Array containing 3*N coordinates, where N is the number of
            atoms.
        box: list of list of float
            Cell object from ASE, which contains the box vectors.
        residue_atom_lists: list of list of int
            list containing the lists of integer indices of atoms
            grouped by residue.

        Returns
        -------
        shift_array: NumPy array
            array containing the shifted positions
        """
        shift_array = np.zeros_like(position_array)
        # Loop through the molecules in the residue list, which is a
        # list containing atom indices.  Molecules are wrapped according
        # to the position of the first atom listed for the residue.
        for residue in self.residue_atom_lists:
            residue_coordinates = position_array[residue]
            displacement_0 = np.subtract(residue_coordinates[0],
                                         residue_coordinates[1:])
            # Assume box sides are all the same length (cubic box).
            diff = (box[0][0] * -np.sign(displacement_0) 
                    * np.floor(abs(displacement_0/box[0][0])+0.5))
            shift_array[residue[1:]] += diff
        return shift_array

    def embed_electrostatics(self, positions, box):
        """
        Collects the indices of atoms which fall within the embedding
        cutoff of the centroid of the QM atoms.

        Parameters
        ----------
        positions: NumPy array
            Array of atom positions within the periodic box
        box: list of list of float
            Cell object from ASE, which contains the box vectors.
        """
        qm_centroid = [sum([positions[i][j] for i in self.qm_atoms_list])
                       / len(self.qm_atoms_list) for j in range(3)]
        embedding_list = []
        qm_drude_list = []
        delr_vector_list = []
        for residue in self.residue_atom_lists:
            # Get the least mirror distance between the QM molecule
            # centroid and the centroid of the current molecule.
            nth_centroid = [sum([positions[i][j] for i in residue]) 
                            / len(residue) for j in range(3)]
            # Legacy embedding.
            nth_centroid = [positions[residue[0]][j] for j in range(3)]
            r_vector = least_mirror_distance(qm_centroid, 
                                             nth_centroid,
                                             box)
            distance = sum([r_vector[i]**2 for i in range(3)])**(0.5)
            if distance < self.embedding_cutoff:
                if not any([atom in self.qm_atoms_list for atom in residue]):
                    embedding_list.append(residue)
                    delr_vector_list.append([r_vector[k]
                                             - nth_centroid[k] for k in range(3)])
                # If atoms are not in the qm_atoms_list and they share 
                # the same residue as the QM atoms, then they must be 
                # drudes from the QM atoms.
                else:
                    qm_drude_list = np.setdiff1d(np.array(residue), 
                                                 np.array(self.qm_atoms_list))
        self.embedding_list = embedding_list
        self.delr_vector_list = delr_vector_list
        self.qm_drude_list = qm_drude_list
