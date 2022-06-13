#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenMM interface to model the MM subsystem of the QM/MM system.
"""
import numpy as np
import openmm.app
import openmm
import simtk.unit
import sys

class OpenMMInterface:
    """
    OpenMM interface for the MM subsytem. 

    Parameters
    ----------
    pdb_file: string
        The directory and filename for the PDB file.
    residue_xml_list: list of str
        The directories and filenames for the XML topology file.
    ff_xml_list: list of str
        The directories and filenames for the XML force field file.
    platform: str
        One of four available platforms on which to run OpenMM.  These 
        include "Reference", "CPU", "CUDA", and "OpenCL".
    temperature: float, Optional, default=300
        The temperature of the system in Kelvin.
    temperature_drude: float, Optional, default=1
        The temperature at which to equilibrate the Drude particles in
        Kelvin.
    friction: float, Optional, default=1
        Determines coupling to the heat bath in Langevin Integrator in
        inverse picoseconds.
    friction_drude: float, Optional, default=1
        Determines coupling to the heat bath during equilibration of the
        Drude particles in inverse picoseconds.
    time_step: float, Optional, default=0.001
        Time step at which to perform the simulation.
    charge_threshold: float, Optional, default=1e-6
        Threshold for charge convergeance.
    nonbonded_cutoff: floar, Optional, default=1.4
        Cutoff for nonbonded interactions, such as Lennard-Jones and 
        Coulomb, in nanometers.
    """
    # Define conversion factors for the Class.
    nm_to_bohr = 18.89726
    hartree_to_kjmol = 2625.4996
    nm_to_angstrom = 10.0
    angstrom_to_bohr = 1.889726

    def __init__(self, pdb_file, residue_xml_list, ff_xml_list, platform,
                 temperature=300, temperature_drude=1, friction=1, friction_drude=1,
                 time_step=0.001, charge_threshold=1e-6, nonbonded_cutoff=1.4):
        # If these options differ from the options provided to the 
        # QMMMEnvironment object, then the QMMMEnvironment options will
        # supercede these.
        self._temperature = temperature * simtk.unit.kelvin
        self._temperature_drude = temperature_drude * simtk.unit.kelvin
        self._friction = friction / simtk.unit.picosecond
        self._friction_drude = friction_drude / simtk.unit.picosecond
        self._time_step = time_step * simtk.unit.picoseconds
        self._charge_threshold = charge_threshold
        self._nonbonded_cutoff = nonbonded_cutoff * simtk.unit.nanometer
        # Load bond definitions before creating pdb object (which calls
        # createStandardBonds() internally upon __init__).  Note that
        # loadBondDefinitions is a static method of Topology, so even
        # though PDBFile creates its own topology object, these bond
        # definitions will be applied.
        for residue_xml in residue_xml_list:
            openmm.app.topology.Topology().loadBondDefinitions(residue_xml)
        self.pdb = openmm.app.pdbfile.PDBFile(pdb_file)
        self.modeller = openmm.app.modeller.Modeller(self.pdb.topology, 
                                                     self.pdb.positions)
        self.forcefield = openmm.app.forcefield.ForceField(*ff_xml_list)
        # Compiling lists of non-drude atoms for each residue of the
        # topology: residue_atom_lists is grouped by residues, and 
        # all_atoms_list is not grouped by residue.
        residue_atom_lists = []
        all_atoms_list = []
        for residue in self.pdb.topology.residues():
            atom_list = []
            for atom in residue._atoms:
                atom_list.append(atom.index)
                all_atoms_list.append(atom.index)
            residue_atom_lists.append(atom_list)
        self._residue_atom_lists = residue_atom_lists
        self._all_atoms_list = all_atoms_list
        self._qm_atoms_list = None
        self._mm_atoms_list = None
        # Add extra particles, such as Drudes or virtual sites.
        self.modeller.addExtraParticles(self.forcefield)
        # If we've added any Drude particles, then this simulation is
        # polarizable.
        self.polarization = True
        if self.pdb.topology.getNumAtoms() == self.modeller.topology.getNumAtoms():
            self.polarization = False
        self._platform = platform

    def setup_system_lj_forces(self):
        """
        this sets up the force classes for QM/MM mechanical embedding,
        which is done within system_lj

        essentially, all irrelevant force classes are removed from the system
        object, and the mechanical embedding is done by creating a
        customnonbonded force with interaction groups between QM/MM
        """
        # first, remove all forces from system_lj ..
        N_Forces = self.system_lj.getNumForces()
        for i in range(N_Forces):
            self.system_lj.removeForce(0)
        # Interaction groups require atom indices to be input as sets.
        qm_atoms_list = set(self._qm_atoms_list)
        mm_atoms_list = set(self._mm_atoms_list)
        # get sigma and epsilon LJ parameters for mechanical embedding
        # if there is a CustomNonbondedForce and these are not in the NonbondedForce,
        # code will have crashed out before call to this subroutine
        sigma_list = []
        eps_list = []
        for i in self._all_atoms_list:
            try:
                (q, sig, eps) = self.nonbonded_force.getParticleParameters(i)
            except:
                print("Something went wrong with index %s" % i)
            sigma_list.append(sig._value)
            eps_list.append(eps._value)
        # create a new CustomNonbondedForce for the mechanical embedding
        custom_nonbonded_force = openmm.CustomNonbondedForce(
            """4*epsilon*((sigma/r)^12-(sigma/r)^6);
             sigma=0.5*(sigma1+sigma2);
             epsilon=sqrt(epsilon1*epsilon2)""")
        custom_nonbonded_force.addPerParticleParameter('epsilon')
        custom_nonbonded_force.addPerParticleParameter('sigma')
        custom_nonbonded_force.setNonbondedMethod(min(self.nonbonded_force.getNonbondedMethod(),
                                                               openmm.NonbondedForce.CutoffPeriodic))
        # add particles with LJ parameters
        for i in self._all_atoms_list:
            custom_nonbonded_force.addParticle([eps_list[i], sigma_list[i]])
        # Mechanical embedding:  Interaction only between QM and MM atoms
        custom_nonbonded_force.addInteractionGroup(qm_atoms_list, mm_atoms_list)
        # Add force to system.
        self.system_lj.addForce(custom_nonbonded_force)


    def create_subsystem(self):
        """
        Create the MM subsystem.

        This is acheived by creating two simulation object.  One will
        include all interactions, as in a standard MD simulation, and
        the other will only include Lennard-Jones interactions between
        the QM atoms and the MM atoms.
        """
        if self.polarization:
            # Polarizable simulation uses Drude integrator with standard
            # settings.
            self.integrator_mm = openmm.DrudeLangevinIntegrator(self._temperature,
                                                                self._friction,
                                                                self._temperature_drude,
                                                                self._friction_drude,
                                                                self._time_step)
            self.integrator_lj = openmm.DrudeLangevinIntegrator(self._temperature,
                                                                self._friction,
                                                                self._temperature_drude,
                                                                self._friction_drude,
                                                                self._time_step)
            # This should prevent polarization catastrophe during
            # equilibration, but shouldn't affect results afterwards
            # (0.2 Angstrom displacement is very large for equilibrating
            # Drudes).
            self.integrator_mm.setMaxDrudeDistance(0.02)
            self.integrator_lj.setMaxDrudeDistance(0.02)
        else:
            # Non-polarizable simulation integrator.
            self.integrator_mm = openmm.LangevinIntegrator(self._temperature,
                                                           self._friction,
                                                           self._time_step)
            self.integrator_lj = openmm.LangevinIntegrator(self._temperature,
                                                           self._friction,
                                                           self._time_step)
        # Create openMM system objects.
        self.system_mm = self.forcefield.createSystem(self.modeller.topology,
                                                      nonbondedCutoff=self._nonbonded_cutoff,
                                                      constraints=None,
                                                      rigidWater=False)
        self.system_lj = self.forcefield.createSystem(self.modeller.topology,
                                                      nonbondedCutoff=self._nonbonded_cutoff,
                                                      constraints=None,
                                                      rigidWater=False)
        # only save force objects internally for system_mm , we don't need to store force objects for system_lj
        # which will only have a customnonbondedforce after modification
        self.nonbonded_force = [f for f in [self.system_mm.getForce(i)
                                for i in range(self.system_mm.getNumForces())]
                                if type(f) == openmm.NonbondedForce][0]
        self.custom_nonbonded_force = [f for f in [self.system_mm.getForce(i)
                                       for i in range(self.system_mm.getNumForces())]
                                       if type(f) == openmm.CustomNonbondedForce]
        if not self.custom_nonbonded_force:
            self.custom_nonbonded_force = None
        else:
           # crash out if CustomNonbondedForce exists, because setup_system_lj_forces cannot handle this ...
           print( 'need to modify setup_system_lj_forces() in code in order to run simulation with a CustomNonbondedForce !')
           sys.exit()
        #  Set long-range interaction method, maybe fine to hard code these settings in as these should be used in anything but test cases ...
        self.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        #self.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
        if self.custom_nonbonded_force:
            self.custom_nonbonded_force.setNonbondedMethod(min(self.nonbonded_force.getNonbondedMethod(),
                                                               openmm.NonbondedForce.CutoffPeriodic))
        # Setup system_lj with appropriate forces for mechanical embedding
        self.setup_system_lj_forces()
        self.platform = openmm.Platform.getPlatformByName(self._platform)
        self.simmd_mm = openmm.app.Simulation(self.modeller.topology,
                                              self.system_mm,
                                              self.integrator_mm,
                                              self.platform)
        self.simmd_mm.context.setPositions(self.modeller.positions)
        self.simmd_lj = openmm.app.Simulation(self.modeller.topology,
                                              self.system_lj,
                                              self.integrator_lj,
                                              self.platform)
        self.simmd_lj.context.setPositions(self.modeller.positions)
        # Remove double-counted intramolecular interactions for QM subsystem.
        self.harmonic_bond_force = [f for f in [self.system_mm.getForce(i)
                                    for i in range(self.system_mm.getNumForces())]
                                    if type(f) == openmm.HarmonicBondForce][0]
        self.harmonic_angle_force = [f for f in [self.system_mm.getForce(i)
                                     for i in range(self.system_mm.getNumForces())]
                                     if type(f) == openmm.HarmonicAngleForce][0]
        for i in range(self.harmonic_bond_force.getNumBonds()):
            p1, p2, r0, k = self.harmonic_bond_force.getBondParameters(i)
            if p1 in self._qm_atoms_list or p2 in self._qm_atoms_list:
                k = simtk.unit.Quantity(0, unit=k.unit)
                self.harmonic_bond_force.setBondParameters(i, p1, p2, r0, k)
        self.harmonic_bond_force.updateParametersInContext(self.simmd_mm.context)
        for i in range(self.harmonic_angle_force.getNumAngles()):
            p1, p2, p3, r0, k = self.harmonic_angle_force.getAngleParameters(i)
            if p1 in self._qm_atoms_list or p2 in self._qm_atoms_list or p3 in self._qm_atoms_list:
                k = simtk.unit.Quantity(0, unit=k.unit)
                self.harmonic_angle_force.setAngleParameters(i, p1, p2, p3, r0, k)
        self.harmonic_angle_force.updateParametersInContext(self.simmd_mm.context)
        # Set force groups for system_mm
        for i in range(self.system_mm.getNumForces()):
            f = self.system_mm.getForce(i)
            f.setForceGroup(i)
        # Set force groups for system_lj
        for i in range(self.system_lj.getNumForces()):
            f = self.system_lj.getForce(i)
            f.setForceGroup(i) 
       # store masses/charges
        masses = []
        charges = []
        for i in range(self.system_mm.getNumParticles()):
            if self.system_mm.getParticleMass(i)/simtk.unit.dalton > 1.0:
                masses.append(self.system_mm.getParticleMass(i)/simtk.unit.dalton)
                (q, sig, eps) = self.nonbonded_force.getParticleParameters(i)
                charges.append(q._value)
        self._masses = masses
        self._charges = charges
        # Drude support will be added in future versions of the code.
        #if self.polarization:
        #    self.generate_drude_pairs()

    #def generate_drude_pairs(self):
    #    """
    #    Setting the pairings between drudes and their parent atoms
    #    """
    #    drude_force = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
    #    self.drude_pairs = []
    #    for i in range(drude_force.getNumParticles()):
    #        particles = drude_force.getParticleParameters(i)
    #        self.drude_pairs.append((particles[0], particles[1]))

    def initial_positions(self, positions):
        """
        Sets initial positions of the OpenMM Modeller.

        Parameters
        ----------
        positions: Numpy array
            Array of atom positions from the ASE Atoms object.
        """
        self._positions = positions
        self.positions = []
        for i in range(len(self._positions)):
            # Update pdb positions.
            self.positions.append(openmm.Vec3(self._positions[i][0]/self.nm_to_angstrom,
                                              self._positions[i][1]/self.nm_to_angstrom,
                                              self._positions[i][2]/self.nm_to_angstrom,
                                             ) * simtk.unit.nanometer)
        # Update positions in modeller object.
        #self.modeller = openmm.app.Modeller(self.pdb.topology, self.positions)
        # Add dummy site and shell initial positions.
        #self.modeller.addExtraParticles(self.forcefield)
        self.simmd_mm.context.setPositions(self.positions)
        self.simmd_lj.context.setPositions(self.positions)

    def set_positions(self, positions):
        """
        Sets positions of atoms within the OpenMM simulation.

        Parameters
        ----------
        positions: Numpy array
            Array of atom positions from the ASE Atoms object.
        """
        # OpenMM requires coordinates in nanometers, whereas the ASE
        # Atoms object provides coordinates in Angstroms.
        self._positions = positions
        self.positions = []
        initial_positions = self.simmd_mm.context.getState(getPositions=True).getPositions()
        for i in range(len(self._positions)):
            self.positions.append(openmm.Vec3(self._positions[i][0]/self.nm_to_angstrom,
                                              self._positions[i][1]/self.nm_to_angstrom,
                                              self._positions[i][2]/self.nm_to_angstrom,
                                             ) * simtk.unit.nanometer)
        # Support for Drudes will be included in a future version of the
        # repository.
        #if self.polarization:
        #    self.set_drude_displacement(initial_positions)
        self.simmd_mm.context.setPositions(self.positions)
        self.simmd_lj.context.setPositions(self.positions)

    #def set_drude_displacement(self, initial_positions):
    #    """
    #    Sets the displacement for the drudes from their parent atoms
    #    """
    #    for i in range(len(self.drude_pairs)):
    #        displacement = self.positions[self.drude_pairs[i][1]] - initial_positions[self.drude_pairs[i][1]]
    #        self.positions[self.drude_pairs[i][0]] += displacement

    def compute_mm_energy(self):
        """
        Calculates the MM subsubsystem energy and forces.

        Returns
        -------
        mm_energy: Numpy array
            Subsubsystem energy.
        mm_forces: Numpy array
            Forces acting on atoms in the subsystem.
        """
        # Get energy and forces from the state.
        state = self.simmd_mm.context.getState(getEnergy=True,
                                               getForces=True,
                                               getPositions=True)
        mm_energy = np.asarray(state.getPotentialEnergy()
                               / simtk.unit.kilojoule_per_mole)
        mm_forces = np.asarray(state.getForces(asNumpy=True)[self._all_atoms_list]
                               / simtk.unit.kilojoule_per_mole
                               * simtk.unit.nanometers)
        # Forces from simmd_mm (MM atoms only).
        for i in self._qm_atoms_list:
            mm_forces[i] = [0, 0, 0]
        self.mm_energy = mm_energy
        self.mm_forces = mm_forces
        self.simmd_mm.reporters[0].report(self.simmd_mm, state)
        # Uncomment to see breakdown of energy by force group
        #print("MM System Forces:")
        with open(self.logger, "a") as fh:
            category = "OpenMM Energy"
            value = str(mm_energy)
            left, right = value.split(".")
            fh.write(category + ":" + " "*(31-len(left)-len(category)) + left + "." + right[0] + " kJ/mol\n")
            for i in range(self.system_mm.getNumForces()):
                f = self.system_mm.getForce(i)
                category = str(type(f)).split("openmm.openmm.")[1].split("\'")[0]
                value = str(self.simmd_mm.context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()._value)
                left, right = value.split(".")
                fh.write("|_" + category + ":" + " "*(29-len(left)-len(category)) + left + "." + right[0] + " kJ/mol\n")
                #print(type(f), str(self.simmd_mm.context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()), flush=True)

    def compute_lj_energy(self):
        """
        Calculates the Lennard-Jones energy and forces for the QM atoms.

        Returns
        -------
        lj_energy: Numpy array
            Subsubsystem energy.
        lj_forces: NumPy array
            Lennard-Jones forces acting on atoms between the QM atoms
            and the MM atoms.
        """
        # Get energy and forces from the state.
        state = self.simmd_lj.context.getState(getEnergy=True,
                                               getForces=True,
                                               getPositions=True)
        lj_energy = np.asarray(state.getPotentialEnergy()
                               / simtk.unit.kilojoule_per_mole)
        lj_forces = np.asarray(state.getForces(asNumpy=True)[self._all_atoms_list]
                               / simtk.unit.kilojoule_per_mole
                               * simtk.unit.nanometers)
        # Forces from simmd_lj (QM atoms only).
        for i in self._mm_atoms_list:
            lj_forces[i] = [0, 0, 0]
        self.lj_energy = lj_energy
        self.lj_forces = lj_forces
        # Uncomment to see breakdown of energy by force group
        #print("LJ System Forces")
        #for i in range(self.system_lj.getNumForces()):
        #    f = self.system_lj.getForce(i)
        #    print(type(f), str(self.simmd_lj.context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()), flush=True)

    def compute_energy(self):
        """
        Calculates the total energy and forces for the MM subsystem.

        Returns
        -------
        openmm_energy: Numpy array
            Subsystem energy
        openmm_forces: Numpy array
            Forces acting on atoms in the MM subsystem.
        """
        self.compute_mm_energy()
        self.compute_lj_energy()
        # Putting force into kJ/mol/Angstrom for ASE to process.
        openmm_energy = self.mm_energy
        openmm_forces = np.add(self.mm_forces, self.lj_forces) / 10.0
        return openmm_energy, openmm_forces

    def generate_reporter(self, name):
        """
        Creates a trajectory reporter.

        Parameters
        ----------
        name: str
            Name of file to write.
        """
        self.simmd_mm.reporters = []
        self.simmd_mm.reporters.append(
            openmm.app.DCDReporter(name, 2),
        )

    def get_charges(self):
        """
        Gets the charges from atoms in the system.

        Returns
        -------
        charges: list of float
            Partial charges by atom index in proton charge.
        """
        return self._charges

    def set_charges(self, charges):
        """
        Sets the charges for atoms in the system.

        Parameters
        ----------
        charges: list of float
            Partial charges by atom index in proton charge.
        """
        self._charges = charges

    def get_masses(self):
        """
        Gets the masses from atoms in the system.

        Returns
        -------
        masses: list of float
            Masses by atom index in Dalton.
        """
        return self._masses

    def set_masses(self, masses):
        """
        Sets the masses for atoms in the system.

        Parameters
        ----------
        masses: list of float
            Masses by atom index in Dalton.
        """
        self._masses = masses

    def get_residue_atom_lists(self):
        """
        Gets the lists of atoms in the system grouped by residue.

        Returns
        -------
        residue_atom_lists: list of int
            Atom indices grouped by residue.
        """
        return self._residue_atom_lists

    def set_residue_atom_lists(self, residue_atom_lists):
        """
        Sets the lists of atoms in the system grouped by residue.

        Parameters
        ----------
        residue_atom_lists: list of int
            Atom Indices grouped by residue.
        """
        self._residue_atom_lists = residue_atom_lists

    def get_qm_atoms_list(self):
        """
        Get the list of QM atom indices.

        Returns
        -------
        qm_atoms_list: list of int
            Indices of the QM atoms within the ASE Atoms object.
        """
        return self._qm_atoms_list

    def set_qm_atoms_list(self, qm_atoms_list):
        """
        Set the list of QM atom indices.

        Parameters
        ----------
        qm_atoms_list: list of int
            Indices of the QM atoms within the ASE Atoms object.
        """
        self._qm_atoms_list = qm_atoms_list
        
    def get_mm_atoms_list(self):
        """
        Get the list of MM atom indices.

        Returns
        -------
        mm_atoms_list: list of int
            Indices of the MM atoms within the ASE Atoms object.
        """
        return self._mm_atoms_list

    def set_mm_atoms_list(self, mm_atoms_list):
        """
        Set the list of MM atom indices.

        Parameters
        ----------
        mm_atoms_list: list of int
            Indices of the MM atoms within the ASE Atoms object.
        """
        self._mm_atoms_list = mm_atoms_list

    def generate_mm_atoms_list(self):
        """
        Create the list of MM subsystem atom indices.
        """
        self._mm_atoms_list = np.setdiff1d(np.array(self._all_atoms_list), 
                                           np.array(self._qm_atoms_list))

    def get_temperature(self):
        """
        Get the MM subsystem temperature.

        Returns
        -------
        temperature: Quantity object
            Temperature of the MM subsystem in Kelvin.
        """
        return self._temperature

    def set_temperature(self, temperature):
        """
        Set the MM subsystem temperature.

        Parameters
        ----------
        temperature: float
            Temperature of the MM subsystem in Kelvin.
        """
        self._temperature = temperature * simtk.unit.kelvin
    
    def get_friction(self):
        """
        Get the MM subsystem friction.

        Returns
        -------
        friction: Quantity object
            Friction of the MM subsystem for the NVT ensemble in inverse
            picoseconds.
        """
        return self._friction

    def set_friction(self, friction):
        """
        Set the MM subsystem friction.

        Parameters
        ----------
        friction: float
            Friction of the MM subsystem for the NVT ensemble in inverse
            picoseconds.
        """
        self._friction = friction / simtk.unit.picosecond

    def get_time_step(self):
        """
        Get the MM subsystem time step.

        Returns
        -------
        time_step: Quantity object
            Time step of the MM subsystem in picoseconds.
        """
        return self._time_step

    def set_time_step(self, time_step):
        """
        Set the MM subsystem time step.

        Parameters
        ----------
        time_step: float
            Time step of the MM subsystem in picoseconds.
        """
        self._time_step = time_step * simtk.unit.picoseconds
