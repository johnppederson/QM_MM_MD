#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for running a SPC/FW water QM/MM/MD simulation.
"""
import sys
sys.path.append("../")
from qm_mm_md import *


def main():
    """
    Run a simulation.
    """
    # Define MM subsystem arguments.
    pdb_file = 'spcfw_box.pdb'
    residue_xml_list = ['spcfw_residues.xml']
    ff_xml_list = ['spcfw.xml']
    platform = 'CPU'
    # Define QM subsystem arguments.
    basis_set = 'STO-3G'
    functional = 'PBE'
    quadrature_radial = 75
    quadrature_spherical = 302
    qm_charge = 0
    qm_spin = 1
    n_threads = 24
    read_guess = False
    # Define QM/MM system arguments.
    qm_atoms_list = [0, 1, 2]
    embedding_cutoff = 9
    name = 'spcfw_nve'
    # Instantiate OpenMM and Psi4 interface objects.
    mm = OpenMMInterface(pdb_file, residue_xml_list, ff_xml_list, platform)
    qm = Psi4Interface(basis_set, functional, quadrature_spherical,
                       quadrature_radial, qm_charge, qm_spin, n_threads,
                       read_guess=read_guess)
    # Instantiate the QM/MM system object.
    qmmm = QMMMEnvironment(pdb_file, './'+name+'_sim_output/', mm, qm,
                           qm_atoms_list, embedding_cutoff)
    qmmm.create_system(name, ensemble="nve", time_step=1, write_freq=1,
                       friction=0.005)
    # Run the simulation.
    qmmm.run_md(10000)

if __name__ == "__main__":
    main()
