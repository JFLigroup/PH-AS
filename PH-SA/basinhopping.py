from ase import *
from ase.io import read
from ase import units
from ase.optimize import BFGS
from ase.optimize.basin import BasinHopping
from deepmd.calculator import DP
import os
import sys
#This is a simple example of basin-hopping-based implementation of configuration exploration
file = sys.argv[1]

atoms = read(file)
atoms.calc = DP(model='OC_10M.pb')
optimizer = BasinHopping(atoms,optimizer=BFGS)
optimizer.run(steps=2)
