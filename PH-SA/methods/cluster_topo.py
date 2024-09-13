import numpy as np
import os
import gudhi
from ase import Atoms
from ase.io import read
from ase.io import Trajectory
from ase.io import write
from ase.neighborlist import natural_cutoffs, NeighborList
from scipy.optimize import least_squares
import math
def objective(x, points, radius):
    return np.linalg.norm(points - x, axis=1) - radius

def calculate_centroid(points,radius):
        #Finding the outer centers of atomic combinations
        initial_guess = np.mean(points, axis=0)
        result = least_squares(objective, initial_guess, args=(points, radius))
        return result.x

def inside_topo(atoms,bond_len,tol=0.6,mul=0.8,radii=5.):
        #Input atoms and bond length for embedding site finding
        pos = atoms.get_positions()
        ac = gudhi.AlphaComplex(points=pos)
        st = ac.create_simplex_tree((radii/2)**2)    
        combinations = st.get_skeleton(9)
        sites = []
        for com in combinations:
                if len(com[0]) >= 2 :
                        temp = pos[com[0]]
                        site = calculate_centroid(temp,math.sqrt(com[1]))
                        sites.append(site)
        rc = gudhi.RipsComplex(points=pos,max_edge_length=radii)
        st = rc.create_simplex_tree(9)
        combinations = st.get_skeleton(9)
        for com in combinations:
                if len(com[0]) > 4  :
                        temp = pos[com[0]]
                        site = calculate_centroid(temp,com[1]/2)
                        sites.append(site)
        latent_sites = []
        for site in sites:
                flag = True
                for atom in pos:
                        if np.linalg.norm(atom - site) < bond_len*mul:
                                flag = False
                                break
                if flag == True:
                        latent_sites.append(site)
        embedding_sites = []
        if tol == False:
               embedding_sites= latent_sites
        #去除重复位点
        else:
                for site in latent_sites:
                        flag  = True
                        for s in embedding_sites:
                                if np.linalg.norm(np.array(s) - np.array(site)) < tol:
                                        flag = False
                        if flag == True:
                                embedding_sites.append(site)
        return embedding_sites

def extend_point_away_from_center(center, point, distance):
    #offset surface sites along the normal vector
    vector = np.array(point) - np.array(center)
    length = np.linalg.norm(vector)
    if length == 0:
        raise ValueError("The point and center cannot be the same.")
    unit_vector = vector / length
    new_point = np.array(center) - distance * unit_vector
    return new_point

def get_surface_atoms_by_coordination(atoms, threshold=10):
    #Access to surface atoms by coordination number
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    surface_atoms = Atoms()
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        coordination_number = len(indices)
        if coordination_number < threshold:
            surface_atoms += atoms[i]
    
    return surface_atoms

def surf_topo(atoms,bond_len,mul=0.8,tol=0.6,radii=5.):
        #Topologizing surface atoms to obtain sites
        center = atoms.get_center_of_mass()
        surf_atoms = get_surface_atoms_by_coordination(atoms)
        surf_atoms.cell = atoms.cell
        surf_pos = surf_atoms.get_positions()
        rc = gudhi.AlphaComplex(points=surf_pos)
        st = rc.create_simplex_tree((radii/2)**2)   
        combinations = st.get_skeleton(9)
        sites= []
        for com in combinations:
                temp = surf_pos[com[0]]
                site = calculate_centroid(temp,math.sqrt(com[1]))
                if len(com[0]) != 1:
                        extended_site = extend_point_away_from_center(site,center,math.sqrt((bond_len*1.15)** 2/(com[1])))
                else:
                        extended_site = extend_point_away_from_center(site,center,bond_len*1.15)
                sites.append(extended_site)
 
        latent_sites = []
        all_pos = atoms.get_positions()
        for si in sites:
                flag = True
                for ap in all_pos:
                        if np.linalg.norm(ap - si)+0.01 < bond_len*mul:
                                flag = False
                                break
                if flag == True:
                        latent_sites.append(si) 
        surface_sites = []
        if tol == False:
                surface_sites = latent_sites   
        else:
                for site in latent_sites:
                        flag  = True
                        for s in surface_sites:
                                if np.linalg.norm(np.array(s) - np.array(site)) < tol:
                                        flag = False
                        if flag == True:
                                surface_sites.append(site)    
        return surface_sites

