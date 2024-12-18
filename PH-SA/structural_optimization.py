from ase.io import read,write,Trajectory 
from deepmd.calculator import DP
from ase.optimize import BFGS
import os
import sys
param1 = sys.argv[1]
file_dir = param1


if __name__ == 'main':
    atoms = read(file_dir)
    atoms.calc = DP(model='./OC_10M.pb')
    dyn = BFGS(atoms)
    dyn.run()
