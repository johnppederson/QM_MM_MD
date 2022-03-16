# QM/MM/MD

This package implements forces and dynamics for the QM/MM/PME method
described by John Pederson and Professor Jesse McDaniel:

DOI:10.1063/5.00xxxxx


## Installation

This software depends on OpenMM, Psi4, ASE, and NumPy.


## Usage

Any python run file should include this directory in its path in order
to use this software.  Required input files include PDB, topology XML,
and forcefield XML for the OpenMM interface.  All other options may be
passed as options in the instantiation of the OpenMMInterface,
Psi4Interface, and QMMMEnvironment objects.


## Authors

Shahriar Khan
Jesse McDaniel
John Pederson
