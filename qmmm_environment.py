from ase import units, Atoms
from ase.io import read, write
from ase.io import Trajectory
from ase.calculators.calculator import Calculator, all_changes
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import QuasiNewton
from copy import deepcopy
import numpy as np
import sys, os, shutil

class Hamiltonian(Calculator):
    """ 
    ASE Calculator. Modeled after SchNetPack calculator.
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/interfaces/ase_interface.py
    """

    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(self, openmm_interface, psi4_interface, qm_atoms_list, embedding_cutoff, residue_list, **kwargs):
        """
        Parameters
        ------------
        openmm_interface : OpenMMInterface object
            OpenMMInterface object containing all the needed info for getting forces from OpenMM
        psi4_interface : Psi4Interface object
            Psi4Interface object containing all the needed info for getting forces from Psi4
        qm_atoms_list : list of integers
            List containing the integer indices of the QM atoms
        embedding_cutoff : float
            Cutoff distance, in Angstroms, within which molecules will be electrostatically embedded in the QM calculation
        residue_list : list
            Residue list containing list of atom indices in each residue
        **kwargs : dict
            Additional args for ASE base calculator
        """

        Calculator.__init__(self, **kwargs)

        self.openmm_interface = openmm_interface
        self.psi4_interface = psi4_interface
        self.qm_atoms_list = qm_atoms_list
        self.embedding_cutoff = embedding_cutoff
        self.residue_list = residue_list

        #self.has_periodic_box = self.openmm_interface.has_periodic_box
        self.has_periodic_box = True
        self.energy_units = units.kJ / units.mol
        self.forces_units = units.kJ / units.mol / units.Angstrom
        self.frame = 0

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Obtains the total energy and force using the above methods

        Parameters
        ------------
        atoms : ASE Atoms object
            atoms object containing coordinates
        properties : list
            Not used, follows SchNetPack format
        system_changes : list
            List of changes for ASE
        """

        result = {}
        if self.has_periodic_box:
            atoms.wrap()
            shift_array = self.make_molecule_whole(atoms.get_positions(), atoms.get_cell())
            atoms.positions -= shift_array

        self.get_embedding_list(atoms.positions, atoms.get_cell())
        self.psi4_interface.set_geometry(self.embedding_list, self.delr_vector_list, atoms.positions, atoms.charges, atoms.get_chemical_symbols())

        Calculator.calculate(self, atoms)

        self.openmm_interface.set_positions(atoms.get_positions()) 
        openmm_energy, openmm_forces = self.openmm_interface.compute_energy()

        qm_energy, qm_forces = self.psi4_interface.compute_energy()

        self.frame += 1
        result["energy"] = openmm_energy.reshape(-1) * self.energy_units
        result["forces"] = openmm_forces.reshape((len(atoms), 3)) * self.forces_units

        print(qm_energy)

        #os.write(0,str.encode('Iteration: '+str(self.frame)+'\n'))

        self.results = result

    def make_molecule_whole(self, position_array, box):
        """
        Atoms are wrapped to stay inside of the periodic box. 
        This function ensures molecules are not broken up by 
        a periodic boundary, as OpenMM electrostatics will be 
        incorrect if all atoms in a molecule are not on the 
        same side of the periodic box. Assumes isotropic box

        Parameters
        ------------
        position_array : NumPy array
            Array containing 3*N coordinates
        box : ASE cell object
            Cell object from ASE, which contains the box vectors
        residue_list : list
            list containing the indices of the atoms of the monomer
            which the reacting atom is contained in

        Returns
        ------------
        shift_array : NumPy array
            array containing the shifted positions
        """

        shift_array = np.zeros_like(position_array)

        for molecule in self.residue_list:
            molecule_coordinates = position_array[molecule]
            disp_0 = np.subtract(molecule_coordinates[0], molecule_coordinates[1:])
            #Assumes box sides are all the same length
            diff = box[0][0] * -np.sign(disp_0) * np.floor(abs(disp_0/box[0][0])+0.5)
            shift_array[molecule[1:]] += diff

        return shift_array

    def get_embedding_list(self, positions, box):
        """
        Collects the indices of atoms which fall within the embedding cutoff of the centroid of the QM atoms

        Parameters
        ------------
        positions : NumPy array
            Array of atom positions within the periodic box
        box : list of lists of floats
            List of lists of floats
        """

        # converting Angstrom cutoff to nm
        embedding_cutoff = self.embedding_cutoff

        qm_centroid = [sum([positions[i][j] for i in self.qm_atoms_list]) / len(self.qm_atoms_list) for j in range(3)]

        # collecting electrostatic_embedding_list and qm_drude_list
        embedding_list = []
        qm_drude_list = []
        delr_vector_list = []

        for atom_list in self.residue_list:

            # getting the least mirror distance between the QM molecule centroid and the centroid of the 
            nth_centroid = [sum([positions[i][j] for i in atom_list]) / len(atom_list) for j in range(3)]
            nth_centroid = [positions[atom_list[0]][j] for j in range(3)]
            r_vector = self.get_least_mirror_distance(qm_centroid, nth_centroid, box)

            distance = sum([r_vector[i]**2 for i in range(3)])**(0.5)

            if distance < embedding_cutoff: 
                if not any([atom in self.qm_atoms_list for atom in atom_list]):
                    embedding_list.append(atom_list)
                    delr_vector_list.append([r_vector[k] - nth_centroid[k] for k in range(3)])

                # if atoms are not in the qm_atoms_list and they share the same residue, then they must be drudes from the QM atoms
                else:
                    qm_drude_list = np.setdiff1d(np.array(atom_list), np.array(self.qm_atoms_list))

        self.embedding_list = embedding_list
        self.delr_vector_list = delr_vector_list
        #self.qm_drude_list = qm_drude_list

    def get_least_mirror_distance(self, i_vector, j_vector, box):
        """
        Returns the least mirror convention distance between i_vec and j_vec

        Parameters
        ------------
        i_vector : NumPy array
            First position vector
        j_vector : NumPy array
            Second position vector
        box_vector : list of lists of floats
            List of lists of floats representing the vectors which describe the principal box

        Returns
        ------------
        r_vector : NumPy array
            Least mirror path between the first and second position vector
        """

        r_vector = [j_vector[k] - i_vector[k] for k in range(3)]
        r_vector -= box[2]*np.floor(r_vector[2]/box[2][2]+0.5)
        r_vector -= box[1]*np.floor(r_vector[1]/box[1][1]+0.5)
        r_vector -= box[0]*np.floor(r_vector[0]/box[0][0]+0.5)

        return r_vector

class QMMMEnvironment:
    """
    Setups and runs the QM/MM/MD simulation. Serves as an interface to OpenMM, Psi4 and ASE.
    Based off of the ASE_MD class from SchNetPack
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/interfaces/ase_interface.py
    """

    def __init__(self, ase_atoms, tmp, openmm_interface, psi4_interface, qm_atoms_list, embedding_cutoff, **kwargs):
        """
        Parameters
        -----------
        ase_atoms : str
            Location of input structure, gets created to ASE Atoms object.
        tmp : str
            Location for tmp directory.
        """

        # setting working directory
        self.working_dir = tmp
        self.tmp = tmp
        if not os.path.isdir(self.tmp):
            os.makedirs(self.tmp)

        # defining Atoms object
        if isinstance(ase_atoms, Atoms):
            self.mol = ase_atoms

        else:

            # reading in PDB file if string
            self.mol = read(ase_atoms)

        # collecting OpenMM interface
        self.openmm_interface = openmm_interface
        self.mol.set_masses(self.openmm_interface.masses)
        self.mol.charges = self.openmm_interface.get_charges()

        # passing positions from Atoms object to OpenMM
        self.openmm_interface.set_initial_positions(self.mol.get_positions())

        # collecting residues from OpenMM
        residue_list = self.openmm_interface.residue_atom_lists
        self.rewrite = kwargs.get('rewrite_log', True)

        # collecting Psi4 interface
        self.psi4_interface = psi4_interface
        self.psi4_interface.set_qm_atoms_list(qm_atoms_list)

        # setting QM atoms list
        self.qm_atoms_list = qm_atoms_list
        self.embedding_cutoff = embedding_cutoff

        # setting calculator
        calculator = Hamiltonian(self.openmm_interface, self.psi4_interface, self.qm_atoms_list, self.embedding_cutoff, residue_list)
        self.mol.set_calculator(calculator)

        self.md = False

    def create_system(self, name, time_step=1.0, temp=300, temp_init=None, restart=False, store=1, nvt=False, friction=0.001):
        """
        Creates the simulation environment for ASE

        Parameters
        ------------
        name : str
            Name for output files.
        time_step : float, optional
            Time step in fs for simulation.
        temp : float, optional
            Temperature in K for NVT simulation.
        temp_init : float, optional
            Optional different temperature for initialization than thermostate set at.
        restart : bool, optional
            Determines whether simulation is restarted or not, 
            determines whether new velocities are initialized.
        store : int, optional
            Frequency at which output is written to log files.
        nvt : bool, optional
            Determines whether to run NVT simulation, default is False.
        friction : float, optional
            friction coefficient in fs^-1 for Langevin integrator
        """

        # setting simulation options for
        if temp_init is None: temp_init = temp
        if not self.md or restart:
            MaxwellBoltzmannDistribution(self.mol, temp_init * units.kB)
            Stationary(self.mol)
            ZeroRotation(self.mol)

        if not nvt:
            self.md = VelocityVerlet(self.mol, time_step * units.fs)
        else:
            self.md = Langevin(self.mol, time_step * units.fs, temperature_K=temp, friction=friction/units.fs)

        logfile = os.path.join(self.tmp, "{}.log".format(name))
        trajfile = os.path.join(self.tmp, "{}.traj".format(name))

        if self.rewrite and os.path.isfile(logfile) and os.path.isfile(trajfile):
            os.remove(logfile)
            os.remove(trajfile)

        logger = MDLogger(self.md, self.mol, logfile, stress=False, peratom=False, header=True, mode="w")
        trajectory = Trajectory(trajfile, "w", self.mol)
        self.md.attach(logger, interval=store)
        self.md.attach(trajectory.write, interval=store)

    def write_mol(self, name, ftype="xyz", append=False):
        """
        Write out current molecule structure.

        Parameters
        ------------
        name : str
            Name of the output file.
        ftype : str, optional
            Determines output file format, default xyz.
        append : bool, optional
            Append to existing output file or not.
        """
        path = os.path.join(self.tmp, "{}.{}".format(name, ftype))
        write(path, self.mol, format=ftype, append=append)

    def calculate_single_point(self):
        """
        Perform a single point computation of the energies and forces and
        store them to the working directory. The format used is the extended
        xyz format. This functionality is mainly intended to be used for
        interfaces.

        Returns
        ------------
        energy : ???
            The energy calculated for the system by ASE
        forces : ???
            The forces calculated for the system by ASE
        """

        # setting OpenMM initial positions
        self.openmm_interface.set_initial_positions(self.mol.get_positions())

        # collecting energy and forces from ASE
        energy = self.mol.get_potential_energy()
        forces = self.mol.get_forces()
        self.mol.energy = energy
        self.mol.forces = forces

        return energy, forces

    def run_md(self, steps):
        """
        Run MD simulation.

        Parameters
        ------------
        steps : int
            Number of MD steps
        """

        self.md.run(steps)

    def optimize(self, fmax=1.0e-2, steps=1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer in ase
        (BFGS + line search)

        Args:
        fmax (float): Maximum residual force change (default 1.e-2)
        steps (int): Maximum number of steps (default 1000)
        """

        name = "optimization"
        optimize_file = os.path.join(self.working_dir, name)
        optimizer = QuasiNewton(
          self.mol,
          trajectory="%s.traj" % optimize_file,
          restart="%s.pkl" % optimize_file,
          )
        optimizer.run(fmax, steps)
     
        # Save final geometry in xyz format
        self.save_molecule(name)
