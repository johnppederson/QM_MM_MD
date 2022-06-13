#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QM/MM system to propogate dynamic simulations.

Manages the interactions between the QM and MM subsytems through their
respective interface objects.
"""
import os
import sys

import numpy as np
import ase
import ase.io
import ase.md
import ase.md.velocitydistribution as ase_md_veldist
import ase.optimize

from .qmmm_hamiltonian import *


class QMMMEnvironment:
    """
    Sets up and runs the QM/MM/MD simulation.  
    
    Serves as an interface to OpenMM, Psi4 and ASE.  Based off of the
    ASE_MD class from SchNetPack.

    Parameters
    ----------
    atoms: str
        Location of input structure to create an ASE Atoms object.
    tmp: str
        Location for tmp directory.
    openmm_interface: OpenMMInterface object
        A pre-existing OpenMM interface with which the QM/MM system may
        communicate.
    psi4_interface: Psi4Interface object
        A pre-existing Psi4 interface with which the QM/MM system may
        communicate.
    qm_atoms_list: list of int
        List of atom indices representing the QM subsystem under
        investigation.
    embedding_cutoff: float
        Cutoff for analytic charge embedding, in Angstroms.
    rewrite_log: bool, Optional, default=True
        Determine whether or not an existing log file gets overwritten.
    """

    def __init__(self, atoms, tmp, openmm_interface, psi4_interface,
                 qm_atoms_list, embedding_cutoff, rewrite_log=True):
        # Set working directory.
        self.tmp = tmp
        if not os.path.isdir(self.tmp):
            os.makedirs(self.tmp)
        # Define Atoms object.
        if isinstance(atoms, ase.Atoms):
            self.atoms = atoms
        else:
            self.atoms = ase.io.read(atoms)
        # Collect OpenMM interface and respective atom properties.
        self.openmm_interface = openmm_interface
        self._residue_atom_lists = self.openmm_interface.get_residue_atom_lists()
        self.openmm_interface.set_qm_atoms_list(qm_atoms_list)
        self.openmm_interface.generate_mm_atoms_list()
        # Collect Psi4 interface and set qm_atoms_list
        self.psi4_interface = psi4_interface
        self.psi4_interface.set_qm_atoms_list(qm_atoms_list)
        self.psi4_interface.set_chemical_symbols(np.asarray(self.atoms.get_chemical_symbols()))
        self.psi4_interface.generate_ground_state_energy(self.atoms.get_positions())
        self.qm_atoms_list = qm_atoms_list
        self.embedding_cutoff = embedding_cutoff
        self.rewrite = rewrite_log

    def create_system(self, name, time_step=1.0, temp=300, temp_init=None,
                      restart=False, write_freq=1, ensemble="nve",
                      friction=0.001, remove_translation=False, 
                      remove_rotation=False):
        """
        Creates the simulation environment for ASE.

        Parameters
        ----------
        name: str
            Name for output files.
        time_step: float, Optional, default=1.0
            Time step in fs for simulation.
        temp: float, Optional, default=300
            Temperature in K for NVT simulation.
        temp_init: float, Optional, default=None
            Optional different temperature for initialization than
            thermostate set at.
        restart: bool, Optional, default=False
            Determines whether simulation is restarted or not, 
            determines whether new velocities are initialized.
        write_freq: int, Optional, default=1
            Frequency at which output is written to log files.  Taken to
            be every x number of time steps.
        ensemble: str, Optional, default="nve"
            Determines which integrator to use given an ensemble of
            variables.
        friction: float, Optional, default=0.001
            Friction coefficient in fs^-1 for Langevin integrator in the
            NVT ensemble.
        remove_translation: bool, Optional, default=False
            Determine whether to zero center of mass translation.
        remove_rotation: bool, Optional, default=False
            Determine whether to zero center of mass rotation.
        """
        # Set initial simulation options.
        if temp_init is None:
            temp_init = temp
        ase_md_veldist.MaxwellBoltzmannDistribution(
            self.atoms,
            temp_init * ase.units.kB,
            rng=np.random.default_rng(seed=42),
        )
        if remove_translation:
            ase_md_veldist.Stationary(self.atoms)
        if remove_rotation:
            ase_md_veldist.ZeroRotation(self.atoms)
        if ensemble.lower() == "nve":
            self.md = ase.md.VelocityVerlet(self.atoms, time_step * ase.units.fs)
        elif ensemble.lower() == "nvt":
            self.md = ase.md.Langevin(self.atoms,
                                      time_step * ase.units.fs,
                                      temperature_K=temp,
                                      friction=friction / ase.units.fs)
        elif ensemble.lower() == "npt":
            print("NPT ensemble is not currently implemented...")
            sys.exit()
        else:
            print("""Unrecognized ensemble input to QMMMEnvironment
                  initialization.""")
            sys.exit()
        # Supercede OpenMMInterface settings.
        self.openmm_interface.set_temperature(temp)
        # ASE takes friction in fs^1, whereas OpenMM takes friction in ps^-1.
        self.openmm_interface.set_friction(friction * 1000)
        # ASE takes time step in fs, whereas OpenMM takes time step in ps.
        self.openmm_interface.set_time_step(time_step / 1000)
        self.openmm_interface.create_subsystem()
        # These are currently implemented only for real atoms, not all
        # particles (such as virtual sites or drudes).
        self.atoms.set_masses(self.openmm_interface.get_masses())
        self.atoms.charges = self.openmm_interface.get_charges()
        self.openmm_interface.initial_positions(self.atoms.get_positions())
        self.psi4_interface.set_charges(self.openmm_interface.get_charges())
        # Define Calculator.
        calculator = QMMMHamiltonian(self.openmm_interface, self.psi4_interface,
                                     self.qm_atoms_list, self.embedding_cutoff,
                                     self._residue_atom_lists)
        self.atoms.set_calculator(calculator)
        # Determine ouput files for simulation.
        log_file = os.path.join(self.tmp, "{}.log".format(name))
        traj_file = os.path.join(self.tmp, "{}.dcd".format(name))
        if (self.rewrite
                and os.path.isfile(log_file)
                and os.path.isfile(traj_file)):
            os.remove(log_file)
            os.remove(traj_file)
        #logger = ase.md.MDLogger(self.md, self.atoms, log_file,
        #                         stress=False, peratom=False, header=True,
        #                         mode="w")
        #self.md.attach(logger, interval=write_freq)
        with open(log_file, "w") as fh:
            fh.write("="*30 + "QM/MM/MD Log" + "="*30 + "\n")
        self.logger = log_file
        self.openmm_interface.logger = log_file
        self.atoms.calc.logger = log_file
        self.openmm_interface.generate_reporter(traj_file)

    def write_atoms(self, name, ftype="xyz", append=False):
        """
        Write out current system structure.

        Parameters
        ----------
        name: str
            Name of the output file.
        ftype: str, Optional, defalt="xyz"
            Determines output file format.
        append: bool, Optional, default=False
            Determine whether to append to existing output file or not.
        """
        path = os.path.join(self.tmp, "{}.{}".format(name, ftype))
        ase.io.write(path, self.atoms, format=ftype, append=append)

    def calculate_single_point(self):
        """
        Perform a single point energy and force computation.

        Returns
        -------
        energy: Numpy array
            The energy calculated for the system by ASE
        forces: Numpy array
            The forces calculated for the system by ASE
        """
        self.openmm_interface.initial_positions(self.atoms.get_positions())
        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        return energy, forces

    def run_md(self, steps):
        """
        Run MD simulation.

        Parameters
        ----------
        steps : int
            Number of MD steps.
        """
        self.md.run(steps)

    def optimize(self, fmax=1.0e-2, steps=1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer.

        Parameters
        ----------
        fmax: float, Optional, default=1.0e-2
            Maximum residual force change.
        steps: int
            Maximum number of steps.
        """
        name = "optimization"
        optimize_file = os.path.join(self.tmp, name)
        optimizer = ase.optimize.QuasiNewton(self.atoms,
                                             trajectory="%s.traj" % optimize_file,
                                             restart="%s.pkl" % optimize_file,)
        optimizer.run(fmax, steps)
        self.write_atoms(name)
