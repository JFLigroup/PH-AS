import numpy as np 
from ase import Atoms
from ase.neighborlist import natural_cutoffs, NeighborList
import gudhi
import math
from scipy.spatial import KDTree
from collections import defaultdict
from ase.data import atomic_numbers, covalent_radii,vdw_radii
import time
from .utils import calculate_centroid , get_cutoffs ,metal_elements,plane_normal
class ClusterAdsorptionSitesFinder():
        """
        Parameters:
                -----------
                atoms : ase.Atoms object
                        The nanoparticle to use as a template to generate surface sites. 
                        Accept any ase.Atoms object. 

                adsorbate_elements : list of strs
                        If the sites are generated without considering adsorbates, 
                        the symbols of the adsorbates are entered as a list, resulting in a clean slab

                bond_len: float , default None
                        The sum of the atomic radii between a reasonable adsorption 
                        substrate and the adsorbate.If not setting, the threshold for 
                        the atomic hard sphere model is automatically calculated using 
                        the covalent radius of ase

                mul : float , default 0.8
                        A scaling factor for the atomic bond lengths used to modulate
                        the number of sites generated, resulting in a larger number of potential sites 

                tol : float , default 0.6
                        The minimum distance in Angstrom between two site for removal
                        of partially overlapping site

                radius : float , default 5.0
                        The maximum distance between two points 
                        when performing persistent homology calculations  

                k : float , default 1.1
                        Expand the key length, so as to calculate the length of the expansion
                        in the direction of the normal vector, k value is too small will lead 
                        to the calculation of the length of the expansion of the error, 
                        you can according to the needs of appropriate increase

                """
        def __init__(self,atoms,
                        adsorbate_elements=[],
                        bond_len = None,
                        mul=0.8,
                        tol=0.7,
                        k=1.1,
                        radius=5.):
                assert True not in atoms.pbc, 'the cell must be non-periodic'
                atoms = atoms.copy()
                for dim in range(3):
                        if np.linalg.norm(atoms.cell[dim]) == 0:
                                atoms.cell[dim][dim] = np.ptp(atoms.positions[:, dim]) + 10.
                self.metal_ids = [a.index for a in atoms if a.symbol not in adsorbate_elements]
                atoms = Atoms(atoms.symbols[self.metal_ids], 
                        atoms.positions[self.metal_ids], 
                        cell=atoms.cell, pbc=atoms.pbc) 
                self.atoms = atoms
                self.bond_len = bond_len
                self.positions = atoms.positions
                self.symbols = atoms.symbols
                self.numbers = atoms.numbers
                self.tol = tol
                self.k = k
                self.mul = mul
                self.radii = radius
                self.cell = atoms.cell
                self.pbc = atoms.pbc
                self.metals = sorted(list(set(atoms.symbols)))
                self.surf_ids, self.surf_atoms = self.get_surface_atoms_by_coordination()
                self.surf_site_list = []
                self.inside_site_list = []
                self.surface_add_radii = covalent_radii[1]
                self.inside_add_radii = covalent_radii[1]
                self.sur_index = []
                self.sites = []
        def get_surface_atoms_by_coordination(self, threshold=10):
                """The function to access to surface atoms by coordination number

                Parameters:
                -----------
                theshold : int , default 10
                        The maximum coordination number of surface atoms.If the surface
                        atoms obtained are not satisfactory, the threshold can be 
                        appropriately mobilized.
                """
                cutoffs = natural_cutoffs(self.atoms)
                nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
                nl.update(self.atoms)
                surface_atoms = Atoms()
                surf_ids = []
                for i in range(len(self.atoms)):
                        indices, offsets = nl.get_neighbors(i)
                        coordination_number = len(indices)
                        if coordination_number < threshold:
                                surf_ids.append(i)
                                surface_atoms += self.atoms[i]
                
                return surf_ids,surface_atoms

        def extend_point_away_from_center(self,center, point, distance):
                """The function is to offset surface sites along the normal vector
                        Generating normal vectors through the center of mass of the initial atom
                Parameters:
                -----------
                center : ndarray 
                The center point of the cluster
                
                points : ndarray
                The site to used for outward offset

                distance : float
                The distance offset in the direction of the normal vector
                """
                vector = np.array(point) - np.array(center)
                length = np.linalg.norm(vector)
                if length == 0:
                        raise ValueError("The point and center cannot be the same.")
                unit_vector = vector / length
                new_point = np.array(center) - distance * unit_vector
                return new_point , unit_vector
        def get_sites(self,absorbent = []):
                if self.sites:
                        return self.sites
                if not self.surf_site_list:
                        self.get_surface_sites(absorbent=absorbent)
                if not self.inside_site_list:
                        self.get_inside_sites(absorbent=absorbent)
                self.sites = self.surf_site_list + self.inside_site_list
                return self.sites

        def surf_topo(self):
                """The function to Topologize surface atoms to obtain sites
                """
                center = self.atoms.get_center_of_mass()
                surf_pos = self.surf_atoms.get_positions()
                surface_atoms = self.surf_atoms
                rc = gudhi.AlphaComplex(points=surf_pos)
                st = rc.create_simplex_tree((self.radii/2)**2)   
                combinations = st.get_skeleton(4)
                sites= []
                site_type = ['top','bridge','hollow',"4fold"]
                fold4_group = []
                del_bri_couple =[]
                # find sites 
                combinations = sorted(list(combinations), key=lambda x: len(x[0]), reverse=True)
                for com in combinations:
                        # Acquisition of surface initial sites
                        temp = surf_pos[com[0]]
                        cov_radii = [covalent_radii[self.surf_atoms[c].number] for c in com[0]]
                        if len(com[0])==1:
                                temp_com = com[0]
                                site = temp[0]
                        elif len(com[0])==2:
                                temp_com = com[0]
                                if tuple(sorted(com[0])) in del_bri_couple:
                                        continue
                                t = cov_radii[1]/sum(cov_radii)
                                site = t * temp[0] + (1 - t) * temp[1]
                        else:
                                if self.bond_len is None:
                                        bond_len = max(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                site = calculate_centroid(temp,cov_radii,math.sqrt(com[1]))
                                temp_com = []
                                cov_radii = []
                                
                                for i,coord in enumerate(surf_pos):
                                        if np.linalg.norm(site - coord) <= bond_len:
                                                temp_com.append(i) 
                                                cov_radii.append(covalent_radii[surface_atoms[i].number])
                                if len(temp_com) == 4:
                                        index_tuple = tuple(temp_com)
                                        if index_tuple in fold4_group:
                                                continue
                                        else:
                                                site = calculate_centroid(surf_pos[temp_com],cov_radii,math.sqrt(com[1]))
                                                fold4_group.append(index_tuple)
                                                max_d = -1
                                                for ind in temp_com[1:]:
                                                        if np.linalg.norm(surf_pos[temp_com[0]]-surf_pos[ind]) > max_d:
                                                                max_d = np.linalg.norm(surf_pos[temp_com[0]]-surf_pos[ind])
                                                                temp_i = ind
                                                del_bri_couple.append((temp_com[0],temp_i))
                                                remain = []
                                                for i in temp_com:
                                                        if i != temp_com[0] and i != temp_i:
                                                                remain.append(i)
                                                del_bri_couple.append(tuple(remain))
                                elif len(temp_com)==3:
                                        temp_com = com[0]
                        #Obtaining the initial site
                        
                        
                        # Determine the metals that make up the site
                        if self.bond_len is None:
                                bond_len = min(cov_radii)+self.surface_add_radii
                        else:
                                bond_len = self.bond_len
                        try:
                                height = math.sqrt((bond_len*self.k)** 2-(com[1]))
                        except Exception as r:
                                height = 0.1
                        
                        site , normal = self.extend_point_away_from_center(site,center,height)

                        # Determine whether the generating site is too close to the site of the initial atom
                        flag = True
                        
                        for ap in self.positions:
                                if np.linalg.norm(ap - site)+0.01 < bond_len*self.mul:
                                        flag = False
                                        break 
                        if flag:
                                sites.append({
                                        'site':site_type[len(temp_com)-1],
                                        'type':'surface',
                                        'normal':normal,
                                        'position':site,   
                                        'indices':[c for c in temp_com]                                    
                                })
                                
                if self.tol == False:
                        self.surf_site_list = sites
                # Determine if the generating sites are too close together
                else:
                        for site in sites:
                                flag  = True
                                for s in self.surf_site_list:
                                        if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                flag = False
                                if flag == True:
                                        self.surf_site_list.append(site) 

        def get_surface_sites(self,absorbent = []):
                if absorbent:
                        self.surface_add_radii = min([covalent_radii[atomic_numbers.get(ele,None)] for ele in absorbent])
                self.surf_topo()
                return self.surf_site_list

        def inside_topo(self):
                """This is a function used to find embedding sites for various 
                non-periodic structures such as nanoclusters
                        
                """
                pos = self.positions
                # Computing sites in 4 dimensions using alpha complexes
                ac = gudhi.AlphaComplex(points=pos)
                st = ac.create_simplex_tree((self.radii/2)**2)    
                combinations = st.get_skeleton(4)
                sites = []
                
                for com in combinations:
                        if len(com[0]) >= 2 :
                                temp = pos[com[0]]
                                cov_radii = [covalent_radii[self.atoms[c].number] for c in com[0]]
                                if len(com[0])==2:
                                        t = cov_radii[1]/sum(cov_radii)
                                        site = t * temp[0] + (1 - t) * temp[1]
                                else:
                                        site = calculate_centroid(temp,cov_radii,math.sqrt(com[1]))
                                if self.bond_len is None:
                                        bond_len = min(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                flag = True
                                for ap in pos:
                                        if np.linalg.norm(ap - site)+0.001 < bond_len*self.mul:
                                                flag = False
                                                break 
                                if flag==True:
                                        sites.append({
                                                'site':'inside',
                                                'type':'inside',
                                                'normal':None,
                                                'position':site,
                                                'indices':[c for c in com[0]]
                                        })                
                # Computing sites larger than 4 dimensions using VR complex shapes
                rc = gudhi.RipsComplex(points=pos,max_edge_length=self.radii)
                st = rc.create_simplex_tree(9)
                combinations = st.get_skeleton(9)
                index_list = []
                for com in combinations:
                        if len(com[0]) > 4 :
                                temp = pos[com[0]]
                                cov_radii = [covalent_radii[self.atoms[c].number] for c in com[0]]
                                site = calculate_centroid(temp,cov_radii,com[1]/2)
                                if self.bond_len is None:
                                        bond_len = min(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                flag = True
                                for ap in pos:
                                        if np.linalg.norm(ap - site)+0.001 < bond_len*self.mul:
                                                flag = False
                                                break 
                                if flag==True:
                                        sites.append({
                                                'site':'inside',
                                                'type':'inside',
                                                'normal':None,
                                                'position':site,
                                                'indices':[c for c in com[0]]
                                        })
                                
                if self.tol == False:
                        self.inside_site_list = sites
                # Determine if the generating sites are too close together
                else:
                        for site in sites:
                                flag  = True
                                for s in self.inside_site_list:
                                        if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                flag = False
                                if flag == True:
                                        self.inside_site_list.append(site) 

        
        def get_inside_sites(self,absorbent = []):
                if absorbent:
                        self.inside_add_radii = min([covalent_radii[atomic_numbers.get(ele,None)] for ele in absorbent])
                self.inside_topo()
                return self.inside_site_list



class SlabAdsorptionsSitesFinder():
        """
                Parameters:
                -----------
                atoms : ase.Atoms object
                        The nanoparticle to use as a template to generate surface sites. 
                        Accept any ase.Atoms object. 

                adsorbate_elements : list of str
                        If the sites are generated without considering adsorbates, 
                        the symbols of the adsorbates are entered as a list, resulting in a clean slab

                bond_len: float
                        The sum of the atomic radii between a reasonable adsorption 
                        substrate and the adsorbate.If not setting, the threshold for 
                        the atomic hard sphere model is automatically calculated using 
                        the covalent radius of ase

                mul : float , default 0.8
                        A scaling factor for the atomic bond lengths used to modulate
                        the number of sites generated, resulting in a larger number of potential sites 

                tol : float , default 0.6
                        The minimum distance in Angstrom between two site for removal
                        of partially overlapping site

                radius : float , default 5.0
                        The maximum distance between two points 
                        when performing persistent homology calculations  

                k : float , default 1.1
                        Expand the key length, so as to calculate the length of the expansion
                        in the direction of the normal vector, k value is too small will lead 
                        to the calculation of the length of the expansion of the error, 
                        you can according to the needs of appropriate increase

                """
        def __init__(self,atoms,
                        adsorbate_elements=[],
                        mul=0.8,
                        tol=0.6,
                        bond_len=None,
                        k=1.1,
                        radius=5.,
                        surface_coordination = 12,
                        both_surface = False
                        ):
                atoms = atoms.copy()
                self.metal_ids = [a.index for a in atoms if a.symbol not in adsorbate_elements]
                atoms = Atoms(atoms.symbols[self.metal_ids], 
                        atoms.positions[self.metal_ids], 
                        cell=atoms.cell, pbc=atoms.pbc) 
                self.atoms = atoms
                self.positions = atoms.positions
                self.symbols = atoms.symbols
                self.numbers = atoms.numbers
                self.tol = tol
                self.k = k
                self.bond_len = bond_len
                self.surface_coordination = surface_coordination
                self.mul = mul
                self.radii = radius
                self.cell = atoms.cell
                self.surface_add_radii = covalent_radii[1] 
                self.inside_add_radii = covalent_radii[1] 
                self.pbc = atoms.pbc
                self.metals = sorted(list(set(atoms.symbols)))
                self.surf_site_list = []
                self.inside_site_list = []
                self.surf_index = []
                self.sites = []
                self.both_surface = both_surface
                self.metal_elements = metal_elements

        
        def get_surface_atoms_by_coordination(self,threshold=12,both_surface=False):
                """The function to access to surface atoms by coordination number

                        Parameters:    
                        -----------

                        theshold : int , default 12
                                The maximum coordination number of surface atoms

                        both_surface , boolen ,default False
                                Whether to return the atoms of the lower surface, 
                                if True, the atoms of the upper and lower surfaces will be returned
                """
                pos = self.positions
                z_coords = pos[:, 2]
                if np.all(z_coords == z_coords[0]) :
                        return range(len(self.atoms)),self.atoms
                cutoffs = get_cutoffs(self.atoms,self.metal_elements)
                nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
                nl.update(self.atoms)
                surface_atoms = Atoms()
                index = []
                for i in range(len(self.atoms)):
                        indices, offsets = nl.get_neighbors(i)
                        coordination_number = len(indices)
                        if coordination_number < threshold:
                                index.append(i)
                                surface_atoms += self.atoms[i]
                z_positions = surface_atoms.positions[:, 2]
                median_z = np.median(z_positions) 
                if both_surface == True:
                        return index , surface_atoms
                upper_index = []
                upper_atoms = Atoms()
                for i , atom in zip(index,surface_atoms):
                        if atom.position[2] > median_z:
                                upper_index.append(i)
                                upper_atoms += atom
                return upper_index , upper_atoms         
        
        def expand_cell(self, cutoff=None, padding=[1,1,0]):

        #Return Cartesian coordinates atoms within a supercell
        #which contains repetitions of the unit cell which contains
        #at least one neighboring atom. Borrowed from Catkit.
                cell = self.atoms.cell
                pbc = [1, 1, 0]
                pos = self.atoms.positions

                if padding is None and cutoff is None:
                        diags = np.sqrt((([[1, 1, 1],
                                        [-1, 1, 1],
                                        [1, -1, 1],
                                        [-1, -1, 1]]
                                        @ cell)**2).sum(1))

                        if pos.shape[0] == 1:
                                cutoff = max(diags) / 2.
                        else:
                                dpos = (pos - pos[:, None]).reshape(-1, 3)
                                Dr = dpos @ np.linalg.inv(cell)
                                D = (Dr - np.round(Dr) * pbc) @ cell
                                D_len = np.sqrt((D**2).sum(1))

                                cutoff = min(max(D_len), max(diags) / 2.)

                latt_len = np.sqrt((cell**2).sum(1))
                V = abs(np.linalg.det(cell))
                # padding = pbc * np.array(np.ceil(cutoff * np.prod(latt_len) /
                                                # (V * latt_len)), dtype=int)

                offsets = np.mgrid[-padding[0]:padding[0] + 1,
                                -padding[1]:padding[1] + 1,
                                -padding[2]:padding[2] + 1].T
                tvecs = offsets @ cell
                coords = pos[None, None, None, :, :] + tvecs[:, :, :, None, :]

                ncell = np.prod(offsets.shape[:-1])
                index = np.arange(len(self.atoms))[None, :].repeat(ncell, axis=0).flatten()
                coords = coords.reshape(np.prod(coords.shape[:-1]), 3)
                offsets = offsets.reshape(ncell, 3)

                return index, coords, offsets
        
        def point_in_range(self,pos):
                """Determine if the site is in the current lattice"""
                z_min = min(self.atoms.positions[:,2])
                z_max = max(self.atoms.positions[:,2])
                cell = self.cell
                for i in range(3): 
                        if not (-0.1 <= pos[i] < cell[i, i]-0.1):
                                return False
                if pos[2]<z_min-0.1 or pos[2]>z_max+0.1:
                        return False
                return True

                        
        def inside_topo(self):
                """ This is a function used to find embedding sites for various
                periodic structures.
                """
                if not self.pbc.all():
                        pos = self.positions
                else:
                        index, pos, offsets = self.expand_cell()
                # Computing sites in 4 dimensions using alpha complexes
                ac = gudhi.AlphaComplex(points=pos)
                st = ac.create_simplex_tree((self.radii / 2) ** 2)    
                combinations = st.get_skeleton(4)
                sites = []
                n = len(self.atoms)
                kdtree = None  # To store the KDTree for fast distance checking
                for com in combinations:
                        if len(com[0]) >= 2 and sorted([c % n for c in com[0]]) not in self.surf_index:
                                temp = pos[com[0]]
                                cov_radii = [covalent_radii[self.atoms[c%n].number] for c in com[0]]
                                if len(com[0])==2:
                                        t = cov_radii[1]/sum(cov_radii)
                                        site = t * temp[0] + (1 - t) * temp[1]
                                else:
                                        site = calculate_centroid(temp,cov_radii,math.sqrt(com[1]))

                                if not self.point_in_range(site):
                                        continue
                                
                                if self.bond_len is None:
                                        bond_len = min(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len

                                # Create a KDTree once, and use it to check proximity for all points
                                if kdtree is None:
                                        kdtree = KDTree(pos)

                                # Check if any point is within bond_len range
                                nearby_points = kdtree.query_ball_point(site, bond_len * self.mul + 0.02)
                                if len(nearby_points) == 0 :
                                        sites.append({
                                        'site': 'inside',
                                        'type': 'inside',
                                        'normal': None,
                                        'position': site,
                                        'indices':[c%n for c in com[0]]
                                        })                       
                # Compute loci larger than 4 dimensions using VR complex shapes
                rc = gudhi.RipsComplex(points=pos, max_edge_length=self.radii)
                st = rc.create_simplex_tree(9)
                combinations = st.get_skeleton(9)
                combinations = sorted(list(combinations), key=lambda x: len(x[0]), reverse=True)
                index_list = []
                for com in combinations:
                        if len(com[0]) > 4:
                                temp = pos[com[0]]
                                cov_radii = [covalent_radii[self.atoms[c%n].number] for c in com[0]]
                                site = calculate_centroid(temp,cov_radii,com[1] / 2)
                                if not self.point_in_range(site):
                                        continue
                                # Calculate the bond length
                                if self.bond_len is None:
                                        bond_len = min(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                # Check proximity using KDTree
                                nearby_points = kdtree.query_ball_point(site, bond_len * self.mul + 0.02)
                                if len(nearby_points) == 0 :
                                        sites.append({
                                        'site': 'inside',
                                        'type': 'inside',
                                        'normal': None,
                                        'position': site,
                                        'indices':[c%n for c in com[0]]
                                        })
                if self.tol is False:
                        for site in sites:
                                flag  = True
                                for s in self.surf_site_list:
                                                if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                        flag = False
                                                        break
                                if flag == True:
                                                self.inside_site_list.append(site) 
                else:
                        for site in sites:
                                flag  = True
                                for s in self.surf_site_list:
                                        if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                flag = False
                                                break
                                if not flag:
                                        continue
                                for s in self.inside_site_list:
                                        if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                flag = False
                                                break
                                if flag == True:
                                        self.inside_site_list.append(site) 
        def expand_surface_cells(self,original_atoms,cell):
                """Return Cartesian coordinates surface_atoms within a supercell
                Parameters:    
                        -----------

                        original_atoms : ase.Atoms object
                                The surface needs to be expand

                        cell , list of float
                               Cells of atomic structure

                """
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

        def calculate_normal_vector(self,positions):
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

        def extend_point_away(self,site,pos,center,height):
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
                        
                        """
                if site[2] >= center[2]:
                        sign = 1
                else:
                        sign = -1
                if len(pos) == 3:
                        normal_vector = self.calculate_normal_vector(pos)
                else:
                        normal_vectors = []
                        index = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]
                        for ind in index:
                               vector = self.calculate_normal_vector(pos[ind])
                               normal_vectors.append(vector) 
                        normal_vector = np.mean(normal_vectors, axis=0)
               
                return  site+normal_vector*height*sign , normal_vector*sign

        def surf_topo(self):
                surface_index , surface_atoms = self.get_surface_atoms_by_coordination(both_surface=self.both_surface,threshold=self.surface_coordination)
                if self.pbc.all() == False:
                        coords = surface_atoms.get_positions()
                else:
                        coords = self.expand_surface_cells(surface_atoms.get_positions(),self.cell)

                rc = gudhi.AlphaComplex(points=coords)
                st = rc.create_simplex_tree((self.radii/2)**2)     
                combinations = st.get_skeleton(4)
                center = self.atoms.get_center_of_mass()
                sites= []
                combinations = sorted(list(combinations), key=lambda x: len(x[0]), reverse=True)
                del_bri_couple = []
                n = len(surface_atoms)
                fold4_group = []
                tri_groups = []
                for com in combinations :
                        if len(com[0])>2:
                                
                                temp = coords[com[0]]
                                cov_radii = [covalent_radii[surface_atoms[c%n].number] for c in com[0]]
                                site = calculate_centroid(temp,cov_radii,math.sqrt(com[1]))
                                if site[2] > max(temp[:,2]) + 0.1:
                                        continue
                                if self.bond_len is None:
                                        bond_len = max(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                temp_com = []
                                cov_radii = []
                                
                                for i,coord in enumerate(coords):
                                        if np.linalg.norm(site - coord) < bond_len*self.k+0.2:
                                                temp_com.append(i)
                                                j = i%len(surface_atoms)
                                                cov_radii.append(covalent_radii[surface_atoms[j].number])
                                if len(temp_com) == 4:
                                        site_type = '4fold'
                                        index_tuple = tuple(temp_com)
                                        if index_tuple in fold4_group:
                                                continue
                                        else:
                                                site = calculate_centroid(coords[temp_com],cov_radii,math.sqrt(com[1]))
                                                fold4_group.append(index_tuple)
                                                max_d = -1
                                                for ind in temp_com[1:]:
                                                        if np.linalg.norm(coords[temp_com[0]]-coords[ind]) > max_d:
                                                                max_d = np.linalg.norm(coords[temp_com[0]]-coords[ind])
                                                                temp_i = ind
                                                del_bri_couple.append((temp_com[0],temp_i))
                                                remain = []
                                                for i in temp_com:
                                                        if i != temp_com[0] and i != temp_i:
                                                                remain.append(i)
                                                del_bri_couple.append(tuple(remain))                      
                                elif len(temp_com)==3:
                                        site_type = 'hollow'
                                        tri_groups.append(temp_com)
                                else:
                                        continue
                                if not self.point_in_range(site):
                                        continue
                                if self.bond_len is None:
                                        bond_len = min(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                try:       
                                        height = math.sqrt((bond_len*self.k)** 2-(com[1]))
                                except Exception as r:
                                        height = 0.1
                                site , normal = self.extend_point_away(site,coords[temp_com],center,height)
                                flag = True
                                for ap in coords:
                                        if np.linalg.norm(ap - site)+0.01 < bond_len*self.mul:
                                                flag = False
                                                break 
                                if flag :
                                        sites.append({
                                                'site':site_type,
                                                'type':'surface',
                                                'normal':normal,
                                                'position':site,
                                                'indices':[c%n for c in temp_com]
                                        })
                                        self.surf_index.append(sorted([surface_index[c%n] for c in com[0]]))
                        if len(com[0])==2:
                                
                                temp = coords[com[0]]
                                if tuple(sorted(com[0])) in del_bri_couple:
                                        continue
                                lam = 2.0
                                for couple in tri_groups:
                                        if com[0][0] in couple and com[0][1] in couple:
                                                lam = 1.0
                                                break             
                                cov_radii = [covalent_radii[surface_atoms[c%n].number] for c in com[0]] 
                                t = cov_radii[1]/sum(cov_radii)
                                site = t * temp[0] + (1 - t) * temp[1]
                                if not self.point_in_range(site):
                                        continue
                                
                                neigh_coords = []
                                for coord in coords:
                                        if np.linalg.norm(site - coord) < sum(cov_radii)*lam:
                                                neigh_coords.append(coord)
                                xyz = np.array(neigh_coords)
                                normal = plane_normal(xyz)
                                center = self.atoms.get_center_of_mass()
                                if site[2] < center[2]:
                                        up = -1
                                else:
                                        up = 1
                                normal *= up  
                                if self.bond_len is None:
                                        bond_len = min(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                try:      
                                        height = math.sqrt((bond_len*self.k)** 2-(com[1]))
                                except Exception as r:
                                        height = 0.1
                                site = site + normal*height
                                flag = True
                                for ap in coords:
                                        if np.linalg.norm(ap - site)+0.01 < bond_len*self.mul:
                                                flag = False
                                                break 
                                if flag :
                                        sites.append({
                                                'site':'bridge',
                                                'type':'surface',
                                                'normal':normal,
                                                'position':site,
                                                'indices':[c%n for c in com[0]]
                                        })
                                        self.surf_index.append(sorted([surface_index[c%n] for c in com[0]]))
                        if len(com[0])==1:
                                temp = coords[com[0]]
                                site = temp[0]
                                if not self.point_in_range(site):
                                        continue
                                metal = surface_atoms[com[0][0]%n].symbol
                                neigh_coords = []
                                for i,coord in enumerate(coords):
                                        neigh_len =covalent_radii[surface_atoms[com[0][0]%n].number] +covalent_radii[surface_atoms[i%n].number]
                                        if np.linalg.norm(site - coord) < neigh_len:
                                                neigh_coords.append(coord)
                                xyz = np.array(neigh_coords)
                                normal = plane_normal(xyz)
                                center = self.atoms.get_center_of_mass()
                                if site[2] < center[2]:
                                        up = -1
                                else:
                                        up = 1
                                normal *= up  
                                if self.bond_len is None:
                                        bond_len = min([covalent_radii[atomic_numbers.get(metal,None)]])+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                height = bond_len*self.k
                                
                                site = site+normal*height
                                flag = True
                                for ap in coords:
                                        if np.linalg.norm(ap - site)+0.01 < bond_len*self.mul:
                                                flag = False
                                                break 
                                if flag :
                                        sites.append({
                                                'site':'top',
                                                'type':'surface',
                                                'normal':normal,
                                                'position':site,
                                                'indices':[c%n for c in com[0]]
                                        })
                if self.tol == False:
                        self.surf_site_list = sites
                # Determine if the generating sites are too close together
                else:
                        for site in sites:
                                flag  = True
                                for s in self.surf_site_list:
                                        if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                flag = False
                                                break
                                if flag == True:
                                        self.surf_site_list.append(site)  

        def get_inside_sites(self,absorbent = []):
                if absorbent:
                        self.inside_add_radii = min([covalent_radii[atomic_numbers.get(ele,None)] for ele in absorbent])
                self.inside_topo()
                return self.inside_site_list
        
        def get_surface_sites(self,absorbent = []):
                if absorbent:
                        self.surface_add_radii = min([covalent_radii[atomic_numbers.get(ele,None)] for ele in absorbent])
                self.surf_topo()
                return self.surf_site_list
        
        def get_sites(self,absorbent = []):
                if self.sites:
                        return self.sites
                if not self.surf_site_list:
                        self.get_surface_sites(absorbent=absorbent)
                if not self.inside_site_list:
                        self.get_inside_sites(absorbent=absorbent)
                self.sites = self.surf_site_list + self.inside_site_list
                return self.sitesa
