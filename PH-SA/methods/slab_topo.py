from ase.io import read,write,Trajectory
import numpy as np
import gudhi
import os
import math
from ase import Atoms
from scipy.optimize import least_squares
from ase.neighborlist import natural_cutoffs, NeighborList
from ase import Atoms 

def expand_cell(atoms, cutoff=None, padding=None , pbc =[1,1,0]):
    """Return Cartesian coordinates atoms within a supercell
    at least one neighboring atom. Modify from Catkit.
    """
    cell = atoms.cell
    cell_copy =  cell.copy()
    pos = atoms.positions
    if padding is None and cutoff is None:
        diags = np.sqrt((([[1, 1, 1],
                           [-1, 1, 1],
                           [1, -1, 1],
                           [-1, -1, 1]]
                           @ cell_copy)**2).sum(1))

        if pos.shape[0] == 1:
            cutoff = max(diags) / 2.
        else:
            dpos = (pos - pos[:, None]).reshape(-1, 3)
            Dr = dpos @ np.linalg.inv(cell_copy)
            D = (Dr - np.round(Dr) * pbc) @ cell_copy
            D_len = np.sqrt((D**2).sum(1))

            cutoff = min(max(D_len), max(diags) / 2.)

    latt_len = np.sqrt((cell_copy**2).sum(1))
    V = abs(np.linalg.det(cell_copy))   
    padding = pbc * np.array(np.ceil(cutoff * np.prod(latt_len) /
                                     (V * latt_len)), dtype=int)
    offsets = np.mgrid[-padding[0]:padding[0] + 1,
                       -padding[1]:padding[1] + 1,
                       -padding[2]:padding[2] + 1].T
    cell_copy[2][2]= max(pos[:,2])
    tvecs = offsets @ cell_copy
    coords = pos[None, None, None, :, :] + tvecs[:, :, :, None, :]
    ncell = np.prod(offsets.shape[:-1])
    index = np.arange(len(atoms))[None, :].repeat(ncell, axis=0).flatten()
    coords = coords.reshape(np.prod(coords.shape[:-1]), 3)
    offsets = offsets.reshape(ncell, 3)
    return index, coords, offsets

def expand_surface_cells(original_atoms,cell):
    """Return Cartesian coordinates surface_atoms within a supercell"""
    la_x = cell[0]
    la_y = cell[1]
    new_atoms = []
    factor = 3
    offset = (factor - 1) / 2 
    offset = int(offset)
    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            for atom in original_atoms:
                new_position = atom + dx*la_x + dy*la_y
                new_atoms.append(new_position)
    
    return np.array(new_atoms)

def point_in_range(pos,atoms):
    """Determine if the site is in the current lattice"""
    cell = atoms.cell
    z_min = min(atoms.positions[:,2])
    z_max = max(atoms.positions[:,2])
    for i in range(3): 
        if not (-0.1 <= pos[i] < cell[i, i]-0.1):
            return False
    if np.linalg.norm(pos[2]-z_min) < 0.1 or np.linalg.norm(pos[2]-z_max) < 0.1:
          return False
    return True

def objective(x, points, radius):
    """This function is to calculate the difference between the distance between two points and the target radius for least squares optimization"""
    return np.linalg.norm(points - x, axis=1) - radius

def calculate_centroid(coordinates,radius):
        """Finding the outer centers of atomic combinations.

        Parameters
        ----------
        points : ndarray
            The Cartesian Cartesian coordinates of atomic combinations.
        
        radius: float
            The distance from outer center to point.
        """
         
        initial_guess = np.mean(coordinates, axis=0)
        result = least_squares(objective, initial_guess, args=(coordinates, radius))
        return result.x

def inside_topo(atoms,bond_len,pbc=True,mul=0.8,tol=0.6,radius=5.):
        """This is a function finding periodic structural 
           embedding sites such as slab
        
        Parameters:
        -----------
        atoms : ase.Atoms object
                The nanoparticle to use as a template to generate embedding sites. 
                Accept any ase.Atoms object. 

        bond_len: float
                The sum of the atomic radii between a reasonable adsorption 
                substrate and the adsorbate.
        pbc : boolen , default True
                Whether or not to expand cells on input atoms to find sites at structural boundaries

        mul : float , default 0.8
                A scaling factor for the atomic bond lengths used to modulate
                the number of sites generated, resulting in a larger number of potential sites 

        tol : float , default 0.6
                The minimum distance in Angstrom between two site for removal
                of partially overlapping site
                
        radius : float , default 5.0
                The maximum distance between two points 
                when performing persistent homology calculations   
                       
        """
        if pbc == False:
              coords = atoms.get_positions()
        else:
              index, coords, offsets = expand_cell(atoms=atoms)
        sites = []
        ac = gudhi.AlphaComplex(points=coords)
        st = ac.create_simplex_tree((radius/2)**2)    
        combinations = st.get_skeleton(9)
        for com in combinations:
                if len(com[0]) >= 2 :
                        temp = coords[com[0]]
                        site = calculate_centroid(temp,math.sqrt(com[1]))
                        sites.append(site)

        rc = gudhi.RipsComplex(points=coords,max_edge_length=radius)
        st = rc.create_simplex_tree(9)   
        combinations = st.get_skeleton(9)
        for com in combinations:
                if len(com[0]) > 4 :
                        temp = coords[com[0]]
                        site = calculate_centroid(temp,com[1]/2.0)
                        sites.append(site)

        sites_in_cell = []
        for site in sites:
                if point_in_range(site,atoms) == True:
                        sites_in_cell.append(site)

        latent_sites = []
        for site in sites_in_cell:
                flag = True
                for atom in coords:
                        if np.linalg.norm(atom - site) < bond_len*mul:
                                flag = False
                                break
                if flag == True:
                        latent_sites.append(site)

        embedding_sites = []
        if tol == False:
               embedding_sites =  latent_sites
        else:
                for site in latent_sites:
                        flag  = True
                        for s in embedding_sites:
                                if np.linalg.norm(np.array(s) - np.array(site)) < tol:
                                        flag = False
                        if flag == True:
                                embedding_sites.append(site)

        return embedding_sites

def calculate_coordination_numbers(atoms):
    #Calculate the atomic coordination number
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    coordination_numbers = []
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        coordination_numbers.append(len(indices))
    
    return coordination_numbers, nl

def calculate_surface_normals(atoms, nl):
    #The function for calculating surface normal vectors to find surface atoms.
    normals = []
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        if len(indices) >= 2:
            vec1 = atoms.positions[indices[0]] - atoms.positions[i]
            vec2 = atoms.positions[indices[1]] - atoms.positions[i]
            normal = np.cross(vec1, vec2)
            normal /= np.linalg.norm(normal)
            normals.append((i, normal))
    return normals

def get_surface_atoms_by_coordination(atoms, threshold=12,both_surface=False):
    """The function to access to surface atoms by coordination number

        Parameters:
        -----------
        atoms : ase.Atoms object
                The nanoparticle to use as a template to find surface atoms. 
                Accept any ase.Atoms object.    

        theshold : int , default 10
                The maximum coordination number of surface atoms

        both_surface , boolen ,default False
                Whether to return the atoms of the lower surface, 
                if True, the atoms of the upper and lower surfaces will be returned
    """
    pos = atoms.get_positions()
    z_coords = pos[:, 2]
    if np.all(z_coords == z_coords[0]) :
           return atoms
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    surface_atoms = Atoms()
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        coordination_number = len(indices)
        if coordination_number < threshold:
            surface_atoms += atoms[i]
    coordination_numbers, nl = calculate_coordination_numbers(surface_atoms)
    z_positions = surface_atoms.positions[:, 2]
    median_z = np.median(z_positions) 
    normals = calculate_surface_normals(surface_atoms, nl)
    upper_surface_indices = []
    lower_surface_indices = []
    for i, normal in normals:
        if normal[2] > 0 and z_positions[i] > median_z:
                upper_surface_indices.append(i)
        elif normal[2] < 0 and z_positions[i] < median_z:
                lower_surface_indices.append(i)
        else:
                if z_positions[i] > median_z:
                        upper_surface_indices.append(i)
                else:
                        lower_surface_indices.append(i)
    upper_surface_atoms = surface_atoms[upper_surface_indices]
    lower_surface_atoms = surface_atoms[lower_surface_indices]
    if both_surface == True:
        return upper_surface_atoms+lower_surface_atoms
    return upper_surface_atoms

def calculate_normal_vector(positions):
        # Calculation of surface normal vectors for outward expansion of site
        vec1 = positions[1] - positions[0]
        vec2 = positions[2] - positions[0]
        normal = np.cross(vec1, vec2)
        if np.linalg.norm(normal) == 0:
               return np.array([0.,0.,1.])
        if normal[2] < 0:
               normal = -normal
        normal /= np.linalg.norm(normal)
        return normal

def extend_point_away(site,pos,surface_atoms,center,height):
       """The function is to offset surface sites along the normal vector
    
        Parameters:
        -----------
        pos : list of ndarray
                The position of the initial atom used to generate the site

        center : ndarray 
                The center point of the slab
        
        site : ndarray
                The site to used for outward offset

        height : float
                The distance offset in the direction of the normal vector
        
        surface_atoms : ase.Atoms object
                The surface_atoms used to generate surface sites. 
                Accept any ase.Atoms object. 
        """
       surface_points = surface_atoms.get_positions()
       if site[2] >= center[2]:
              sign = 1
       else:
              sign = -1
       if len(pos) == 1:
              distances = np.linalg.norm(surface_points - pos[0], axis=1)
              nearest_indices = np.argsort(distances)[:3]
              p = surface_atoms[nearest_indices]
              pos = p.get_positions()
       elif len(pos) == 2:
              distances = np.linalg.norm(surface_points - pos[0], axis=1) + np.linalg.norm(surface_points - pos[1], axis=1)
              nearest_indices = np.argsort(distances)[:3]
              p = surface_atoms[nearest_indices]
              pos = p.get_positions()
              
       normal_vector = calculate_normal_vector(pos)
       return  site + normal_vector*height*sign

def surface_topo(atoms,bond_len,pbc=True,tol=0.6,mul=0.8,both_surface=False,radius=5.):
        """The function to Topologize surface atoms to obtain sites
        
        Parameters:
        -----------
        atoms : ase.Atoms object
                The slab to use as a template to generate surface sites. 
                Accept any ase.Atoms object. 

        bond_len: float
                The sum of the atomic radii between a reasonable adsorption 
                substrate and the adsorbate.
                
        pbc : boolen , default True
                Whether or not to expand cells on input atoms to find sites at structural boundaries

        mul : float , default 0.8
                A scaling factor for the atomic bond lengths used to modulate
                the number of sites generated, resulting in a larger number of potential sites 

        tol : float , default 0.6
                The minimum distance in Angstrom between two site for removal
                of partially overlapping site

        radius : float , default 5.0
                The maximum distance between two points 
                when performing persistent homology calculations  
        
        both_surface , boolen ,default False
                Whether to return the sites of the lower surface, 
                if True, the sites of the upper and lower surfaces will be returned

        """
       
        surface_atoms = get_surface_atoms_by_coordination(atoms,both_surface=both_surface)
        if pbc == False:
              coords = surface_atoms.get_positions()
        else:
              coords = expand_surface_cells(surface_atoms.get_positions(),atoms.cell)
        center = atoms.get_center_of_mass()
        rc = gudhi.AlphaComplex(points=coords)
        st = rc.create_simplex_tree((radius/2)**2)     
        combinations = st.get_skeleton(9)
        sites= []
        for com in combinations:
                temp = coords[com[0]]
                site = calculate_centroid(temp,math.sqrt(com[1]))
                if len(com[0]) != 1:
                        extended_site = extend_point_away(site,temp,surface_atoms,center,math.sqrt((bond_len*1.15)** 2/((com[1]))))
                else:
                        extended_site = extend_point_away(site,temp,surface_atoms,center,bond_len*1.15)
                sites.append(extended_site)
        sites_in_cell = []
        for site in sites:
                if point_in_range(site,atoms) == True:
                        sites_in_cell.append(site)
        latent_sites = []
        for site in sites_in_cell:
                flag = True
                for atom in coords:
                        if np.linalg.norm(atom - site) < bond_len*mul:
                                flag = False
                                break
                if flag == True:
                        latent_sites.append(site)

        surface_sites = []
        if tol == False:
               surface_sites =  latent_sites
        else:
                for site in latent_sites:
                        flag  = True
                        for s in surface_sites:
                                if np.linalg.norm(np.array(s) - np.array(site)) < tol:
                                        flag = False
                        if flag == True:
                                surface_sites.append(site)

        return surface_sites
