from openmm.app import *
from openmm import *
from simtk.unit import *
import numpy as np
import sys

class OpenMMInterface:
    """
    Base system being defined for both Psi4 and OpenMM; modeled after MM_base class:
    https://github.com/jmcdaniel43/MM_base/blob/ee01e937a8d2fd146930a3237a6ac7837d78774c/MM_class_base.py
    """

    def __init__(self, pdb_file, residue_xml_list, ff_xml_list, platform, **kwargs):
        """
        Parameters
        ------------
        pdb_file : string
            String continaing the directory and filename for the PDB file modeling the system
        residue_xml_list : list
            List of strings containing the directories and filenames for topology
        ff_xml_list : list
            List of strings containing the directories and filenames for the force field
        platform : str
            One of four available platforms on which to run OpenMM
        **kwargs : dict
            Dictionary of values to over-ride the default simulation parameters
        """

        # conversion factors
        self.nm_to_bohr = 18.89726
        self.hartree_to_kjmol = 2625.4996
        self.nm_to_angstrom = 10.0
        self.angstrom_to_bohr = 1.889726

        # default run parameters; input in **kwargs may overide defaults
        self.temperature = 300*kelvin
        self.temperature_drude = 1*kelvin
        self.friction = 1/picosecond
        self.friction_drude = 1/picosecond
        self.timestep = 0.001*picoseconds
        self.small_threshold = 1e-6  # threshold for charge magnitude
        self.cutoff = 1.4*nanometer

        # reading inputs from **kwargs
        if 'temperature' in kwargs:
            self.temperature = int(kwargs['temperature'])*kelvin
        if 'temperature_drude' in kwargs:
            self.temperature_drude = int(kwargs['temperature_drude'])*kelvin
        if 'friction' in kwargs:
            self.friction = int(kwargs['friction'])/picosecond
        if 'friction_drude' in kwargs:
            self.friction_drude = int(kwargs['friction_drude'])/picosecond
        if 'timestep' in kwargs:
            self.timestep = float(kwargs['timestep'])*picoseconds
        if 'small_threshold' in kwargs:
            self.small_threshold = float(kwargs['small_threshold'])
        if 'cutoff' in kwargs:
            self.cutoff = float(kwargs['cutoff'])*nanometer
        if 'electrostatic_embedding_list' in kwargs:
            self.electrostatic_embedding_list = eval(kwargs['electrostatic_embedding_list'])

        # load bond definitions before creating pdb object (which calls createStandardBonds() internally upon __init__).  Note that loadBondDefinitions is a static method
        # of Topology, so even though PDBFile creates its own topology object, these bond definitions will be applied...
        for residue_file in residue_xml_list:
            Topology().loadBondDefinitions(residue_file)

        # now create pdb object, use first pdb file input
        self.pdb = PDBFile(pdb_file)

        # create modeller
        self.modeller = Modeller(self.pdb.topology, self.pdb.positions)

        # create force field
        self.forcefield = ForceField(*ff_xml_list)

        # compiling list of non-drude atoms for each residue of the topology
        residue_atom_lists = []
        for residue in self.pdb.topology.residues():
            atom_list = []
            for atom in residue._atoms:
                atom_list.append(atom.index)
            residue_atom_lists.append(atom_list)
        self.residue_atom_lists = residue_atom_lists

        all_atom_list = []
        for i in residue_atom_lists:
            all_atom_list.extend(i)

        qm_atoms_list = []

        np.array(qm_atoms_list)
        np.array(all_atom_list)
        mm_atoms_list = np.setdiff1d(all_atom_list, qm_atoms_list)

        #converting to list from ndarray
        mm_atoms_list = mm_atoms_list.tolist()

        self.qm_atoms_list = qm_atoms_list
        self.mm_atoms_list = mm_atoms_list
        self.all_atom_list = all_atom_list
        
        # add extra particles
        self.modeller.addExtraParticles(self.forcefield)

        # polarizable simulation?  Figure this out by seeing if we've added any Drude particles ...
        self.polarization = True
        if self.pdb.topology.getNumAtoms() == self.modeller.topology.getNumAtoms():
            self.polarization = False

        if self.polarization:
            # polarizable simulation, use Drude integrator with standard settings
            self.integrator_mm = DrudeLangevinIntegrator(self.temperature, self.friction, self.temperature_drude, self.friction_drude, self.timestep)
            self.integrator_lj = DrudeLangevinIntegrator(self.temperature, self.friction, self.temperature_drude, self.friction_drude, self.timestep)

            # this should prevent polarization catastrophe during equilibration, but shouldn't affect results afterwards ( 0.2 Angstrom displacement is very large for equil. Drudes)
            self.integrator_mm.setMaxDrudeDistance(0.02)
            self.integrator_lj.setMaxDrudeDistance(0.02)

        else:
            # non-polarizable simulation integrator
            self.integrator_mm = LangevinIntegrator(self.temperature, self.friction, self.timestep)
            self.integrator_lj = LangevinIntegrator(self.temperature, self.friction, self.timestep)

        # create openMM system object
        self.system = self.forcefield.createSystem(self.modeller.topology, nonbondedCutoff=self.cutoff, constraints=None, rigidWater=False)
        self.system_lj = self.forcefield.createSystem(self.modeller.topology, nonbondedCutoff=self.cutoff, constraints=None, rigidWater=False)

        self.platform = Platform.getPlatformByName(platform)
        self.simmd_mm = Simulation(self.modeller.topology, self.system, self.integrator_mm, self.platform)
        self.simmd_mm.context.setPositions(self.modeller.positions)
        self.simmd_lj = Simulation(self.modeller.topology, self.system_lj, self.integrator_lj, self.platform)
        self.simmd_lj.context.setPositions(self.modeller.positions)
        
        # get force types and set method
        self.harmonicBondForce = [f for f in [self.system_lj.getForce(i) for i in range(self.system_lj.getNumForces())] if type(f) == HarmonicBondForce][0]
        self.harmonicAngleForce = [f for f in [self.system_lj.getForce(i) for i in range(self.system_lj.getNumForces())] if type(f) == HarmonicAngleForce][0]
        self.nbondedForce = [f for f in [self.system_lj.getForce(i) for i in range(self.system_lj.getNumForces())] if type(f) == NonbondedForce][0]
        self.customNonbondedForce = [f for f in [self.system_lj.getForce(i) for i in range(self.system_lj.getNumForces())] if type(f) == CustomNonbondedForce][0]
        self.custombond = [f for f in [self.system_lj.getForce(i) for i in range(self.system_lj.getNumForces())] if type(f) == CustomBondForce][0]

        self.qm_atoms_list = set(self.qm_atoms_list)
        self.mm_atoms_list = set(self.mm_atoms_list)

        #cutoff the forces
        for i in range(self.harmonicBondForce.getNumBonds()):
            p1, p2, r0, k = self.harmonicBondForce.getBondParameters(i)
            if p1 in self.all_atom_list or p2 in self.all_atom_list:
                k = Quantity(0, unit=k.unit)
                self.harmonicBondForce.setBondParameters(i, p1, p2, r0, k)
        self.harmonicBondForce.updateParametersInContext(self.simmd_lj.context)

        for i in range(self.harmonicAngleForce.getNumAngles()):
            p1, p2, p3, r0, k = self.harmonicAngleForce.getAngleParameters(i)
            if p1 in self.all_atom_list or p2 in all_atom_list or p3 in self.all_atom_list:
                k = Quantity(0, unit=k.unit)
                self.harmonicAngleForce.setAngleParameters(i, p1, p2, p3, r0, k)
        self.harmonicAngleForce.updateParametersInContext(self.simmd_lj.context)

        if self.custombond:
            for i in range(self.custombond.getNumBonds()):
                p1, p2, parms = self.custombond.getBondParameters(i)
                if p1 not in self.all_atom_list or p2 not in self.all_atom_list:
                    self.custombond.setBondParameters(i, p1, p2, (0.0, 0.1, 10.0))
                p1, p2, parms = self.custombond.getBondParameters(i)
        self.custombond.updateParametersInContext(self.simmd_lj.context)


        sigma_list = []
        eps_list = []
   
        for i in self.all_atom_list:
            try:
                (q, sig, eps) = self.nbondedForce.getParticleParameters(i)
            except:
                print ('Something went wrong with index %s' % i)
            sigma_list.append(sig._value)
            eps_list.append(eps._value)


        for i in self.all_atom_list:
            self.nbondedForce.setParticleParameters(i, 0.0, 0.0, 0.0)
        
        for i in range(self.system_lj.getNumForces()):
            f = self.system_lj.getForce(i)
            type(f)
            f.setForceGroup(i)
       
 
        self.system_lj.removeForce(self.customNonbondedForce.getForceGroup())
       
        self.customNonbondedForce = openmm.CustomNonbondedForce('4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)')

        self.customNonbondedForce.setEnergyFunction('4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)')
        self.customNonbondedForce.addPerParticleParameter('epsilon')
        self.customNonbondedForce.addPerParticleParameter('sigma')
        self.nonbondedMethod = min(self.nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic)
        self.system_lj.removeForce(self.nbondedForce.getForceGroup())

        # add force to system
        self.system_lj.addForce(self.customNonbondedForce)

        for i in self.all_atom_list:
            self.customNonbondedForce.addParticle([eps_list[i], sigma_list[i]])

        self.customNonbondedForce.addInteractionGroup(self.qm_atoms_list, self.mm_atoms_list)

        # Provide new LJ parameters to CustomNonbondedForce
        for i in range(self.customNonbondedForce.getNumParticles()):
            self.customNonbondedForce.setParticleParameters(i, [eps_list[i], sigma_list[i]])

        self.customNonbondedForce.setNonbondedMethod(self.nonbondedMethod)
 

        # we need to reset force groups after adding new customNonbondedForce_alchemical force object
        for i in range(self.system_lj.getNumForces()):
            f = self.system_lj.getForce(i)
            type(f)
            f.setForceGroup(i)
 
        # lastly, reinitialize context to incorporate new alchemical force class
        state = self.simmd_lj.context.getState(getEnergy=False,getForces=True,getVelocities=False,getPositions=True)
        positions = state.getPositions()
        self.simmd_lj.context.reinitialize()
        self.simmd_lj.context.setPositions(positions)

        # get force types and set method
        self.harmonicBondForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == HarmonicBondForce][0]
        self.harmonicAngleForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == HarmonicAngleForce][0]
        self.nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
        self.customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
        self.custombond = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomBondForce][0]

        if self.polarization:
            self.drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]

            # will only have this for certain molecules
            self.custombond = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomBondForce][0]

        # set long-range interaction method
        self.nbondedForce.setNonbondedMethod(NonbondedForce.PME)
        self.customNonbondedForce.setNonbondedMethod(min(self.nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))
 
        # collecting list of non-drude atoms and their masses
        state = self.simmd_mm.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        self.positions = state.getPositions()
        self.atom_list = []
        self.masses = []
        for i in range(self.system.getNumParticles()):
            if self.system.getParticleMass(i)/dalton > 1.0:
                self.atom_list.append(i)
                self.masses.append(self.system.getParticleMass(i)/dalton)
            #self.system.setParticleMass(i,0)

        if self.polarization:
            self.set_drude_pairs()

    def set_drude_pairs(self):
        """
        Setting the pairings between drudes and their parent atoms
        """

        drude_force = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
        self.drude_pairs = []

        for i in range(drude_force.getNumParticles()):
            particles = drude_force.getParticleParameters(i)
            self.drude_pairs.append((particles[0], particles[1]))

    def set_initial_positions(self, position_array):
        """
        Sets initial positions of the OpenMM Modeller using the positions from ASE

        Parameters
        ------------
        position_array : NumpPy array
            Array of atom positions from the ASE
        """

        self.position_array = position_array

        for i in range(len(self.position_array)):

            # update pdb positions
            self.positions[i] = Vec3(self.position_array[i][0]/self.nm_to_angstrom, self.position_array[i][1]/self.nm_to_angstrom, self.position_array[i][2]/self.nm_to_angstrom)*nanometer

        # now update positions in modeller object
        self.modeller = Modeller(self.pdb.topology, self.positions)

        # add dummy site and shell initial positions
        self.modeller.addExtraParticles(self.forcefield)
        self.simmd_mm.context.setPositions(self.modeller.positions)
        self.positions = self.modeller.positions

    def set_positions(self, position_array):
        """
        Sets positions of the OpenMM simulation using the positions determined by the ASE calculator

        Parameters
        ------------
        position_array : NumpPy array
            Array of atom positions from the ASE
        """

        self.position_array = position_array/10
        self.initial_positions = self.simmd_mm.context.getState(getPositions=True).getPositions()

        for i in range(len(self.atom_list)):
            self.positions[self.atom_list[i]] = Vec3(self.position_array[i][0], self.position_array[i][1], self.position_array[i][2])*nanometer

        if self.polarization:
            self.set_drude_displacement()

        self.simmd_mm.context.setPositions(self.positions)

    def set_drude_displacement(self):
        """
        Sets the displacement for the drudes from their parent atoms
        """

        for i in range(len(self.drude_pairs)):
            displacement = self.positions[self.drude_pairs[i][1]] - self.initial_positions[self.drude_pairs[i][1]]
            self.positions[self.drude_pairs[i][0]] += displacement

    def get_charges(self):
        """
        Gets the charges from atoms in the system

        Returns
        ------------
        charge_list : list of floats
            List of partial charges by atom index
        """

        charge_list = []

        for i in self.atom_list:
            (q, sig, eps) = self.nbondedForce.getParticleParameters(i)
            charge_list.append(q._value)

        return charge_list

    def compute_mm_energy(self):
        """
        Calculates the mm energy and forces for ASE to integrate

        Returns
        ------------
        mm_energy : NumPy array
            Array of system energy
        mm_forces : NumPy array
            Array of forces acting on atoms in the system
        """

        # get energy and forces from the state
        state = self.simmd_mm.context.getState(getEnergy=True,getForces=True,getPositions=True)
        self.positions = state.getPositions()
        mm_energy = np.asarray(state.getPotentialEnergy()/kilojoule_per_mole)
        mm_forces = np.asarray(state.getForces(asNumpy=True)[self.atom_list]/kilojoule_per_mole*nanometers)

        for i in self.qm_atoms_list:
            mm_forces[i] = [0, 0, 0]*kilojoule/(nanometer*mole) #forces from simmd_mm (mm atoms only)

        self.mm_energy = mm_energy
        self.mm_forces = mm_forces


    def compute_lj_energy(self):
        """
        Calculates the lj energy and forces for ASE to integrate

        Returns
        ------------
        lj_energy : NumPy array
            Array of system energy
        lj_forces : NumPy array
            Array of forces acting on atoms in the system
        """


        # get energy and forces from the state
        state = self.simmd_lj.context.getState(getEnergy=True,getForces=True,getPositions=True)
        self.positions = state.getPositions()
        lj_energy = np.asarray(state.getPotentialEnergy()/kilojoule_per_mole)
        lj_forces = np.asarray(state.getForces(asNumpy=True)[self.atom_list]/kilojoule_per_mole*nanometers)
	
        self.lj_energy = lj_energy
        self.lj_forces = lj_forces

        # if you want energy decomposition, uncomment these lines...
        #for j in range(self.system.getNumForces()):
        #    f = self.system.getForce(j)
        #    print(type(f), str(self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))


    def compute_energy(self):
        """
        Calculates the openmm energy and forces for ASE to integrate

        Returns
        ------------
        openmm_energy : NumPy array
            Array of system energy
        openmm_forces : NumPy array
            Array of forces acting on atoms in the system
        """
        self.compute_mm_energy()
        self.compute_lj_energy()	


        openmm_energy = np.add(self.mm_energy, self.lj_energy)
        openmm_forces = np.add(self.mm_forces, self.lj_forces)

        return openmm_energy, openmm_forces/10.0


    def get_least_mirror_distance(self, i_vector, j_vector, box_vector):
        """
        Returns the least mirror convention distance between i_vec and j_vec

        Parameters
        ------------
        i_vector : numpy array
            First position vector
        j_vector : numpy array
            Second position vector
        box_vector : list
            List of lists of floats representing the vectors which describe the principal box
        """

        r_vector = [i_vector[k] - j_vector[k] for k in range(3)]
        r_vector -= box_vector[2]*np.floor(r_vector[2]/box_vector[2][2]+0.5)
        r_vector -= box_vector[1]*np.floor(r_vector[1]/box_vector[1][1]+0.5)
        r_vector -= box_vector[0]*np.floor(r_vector[0]/box_vector[0][0]+0.5)

        return r_vector

