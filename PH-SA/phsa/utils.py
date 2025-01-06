import numpy as np
from scipy.optimize import least_squares,minimize
from ase.data import atomic_numbers, covalent_radii,vdw_radii

def objective(x, points,weights, radius):
    """This function is to calculate the difference between the distance between two points and the target radius for least squares optimization"""
    distances = np.linalg.norm(points - x, axis=1)
    return weights * ((distances - radius) ** 2)                    
                                    
def calculate_centroid(points,cov_radii,radius):
    """Finding the outer centers of atomic combinations.

    Parameters
    ----------
    points : ndarray
        The Cartesian Cartesian coordinates of atomic combinations.
    
    radius: float
        The distance from outer center to point.
    """
    weights = cov_radii / np.sum(cov_radii)
    initial_guess = np.mean(points, axis=0)
    result = least_squares(
        lambda x: np.sqrt(objective(x, points, weights, radius)),
        initial_guess,
        method="lm",
        max_nfev=50,
        ftol=1e-4,
    )
    return result.x

def get_cutoffs(atoms,metal_elements,mult=1.1):

        cutoffs = []
        for atom in atoms:
                symbol = atom.symbol  
                if symbol in metal_elements:
                
                        cutoff = covalent_radii[atomic_numbers[symbol]]
                else:
               
                        cutoff = vdw_radii[atomic_numbers[symbol]]
                
               
                if cutoff is None or cutoff <= 0:
                        cutoff = 1.5 
                cutoff *= mult
                cutoffs.append(cutoff)
        return cutoffs

metal_elements = {
        "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", 
        "Ni", "Cu", "Zn", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", 
        "Cd", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", 
        "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", 
        "Hg", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", 
        "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", 
        "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
                }

import random
import numpy as np 
def calculate_distance(cell,point1, point2,pbc):
    """The function to calculated distance 
        for dealing with periodicity

    Parameters:
    -----------
    cell : ndarray
        The cell in input atoms.
    points1 , points2 : ndarray
        The positions to calculate distance with periodicity 
        
    """
    if pbc.any() == True:
        a_vec = cell[0]
        b_vec = cell[1]
        dis = 10.
        neighbors = []
        for da in range(-1,2):
                for db in range(-1,2):
                        neighbor_position =  point1['position'] + da * a_vec + db * b_vec
                        neighbors.append(neighbor_position)
        for n in neighbors:
                if dis > np.linalg.norm(n-point2['position']):
                        dis = np.linalg.norm(n-point2['position'])
        return dis
    else:
        return np.linalg.norm(point1['position']-point2['position'])
    
def calculate_atom_distance(cell,point1, point2,pbc):
    if pbc.any() == True:
        a_vec = cell[0]
        b_vec = cell[1]
        dis = 10.
        neighbors = []
        for da in range(-1,2):
                for db in range(-1,2):
                        neighbor_position =  point1 + da * a_vec + db * b_vec
                        neighbors.append(neighbor_position)
        for n in neighbors:
                if dis > np.linalg.norm(n-point2):
                        dis = np.linalg.norm(n-point2)
        return dis
    else:
        return np.linalg.norm(point1['position']-point2['position'])
    
def select_points(points, num_points, min_distance,cell,pbc):
    """
    The functino is Used to carry out the search for picking sites.

    Parameters:
    -----------
    points : list of ndarray
        All possible potential sites should be entered
    
    num_points: int 
        Number of adsorbents

    min_distance : float 
        The minimum distance between two neighboring adsorbents

    cell : ndarray
        The cell in input atoms.
    """
    selected_points = []
    
    while len(selected_points) < num_points:
        point = random.choice(points)
        if all(calculate_distance(cell,point, selected_point,pbc) >= min_distance for selected_point in selected_points):
            selected_points.append(point)
    return selected_points

from networkx.algorithms import isomorphism
from ase.neighborlist import natural_cutoffs, NeighborList
import networkx as nx
def is_unique(graph,unique_graphs):
    """Determine if the current input graph is unique

    Parameters:
    -----------
    graph : networkx.Graph object
        The graph for determining uniqueness

    unique_graphs: list of networkx.Graph object
        List of saved graphs with unique 
    """
    if unique_graphs==[]:
         return True
    for i,unique_graph in enumerate(unique_graphs):
        GM = isomorphism.GraphMatcher(graph, unique_graph, node_match=isomorphism.categorical_node_match('symbol', ''))
        if GM.is_isomorphic():
            return False
    return True

def get_graph(atoms):
    if atoms.pbc.any() == True:
    #Get Periodicity Graph
        cutoffs = natural_cutoffs(atoms,mult=1.2)
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        G = nx.Graph()
        symbols = atoms.symbols                               
        G.add_nodes_from([(i, {'symbol': symbols[i]}) 
                                for i in range(len(symbols))])
        for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                        distance = calculate_atom_distance(cell=cell,point1=positions[i],point2=positions[j],pbc=atoms.pbc)
                if distance < cutoffs[i]+cutoffs[j]:
                        G.add_edge(i, j)
        graph = wl(G)
        return G
    #Get Nonperiodicity Graph
    else:
        cutoffs = natural_cutoffs(atoms,mult=1.2)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        matrix = nl.get_connectivity_matrix(sparse=False)
        G = nx.Graph()
        symbols = atoms.symbols                               
        G.add_nodes_from([(i, {'symbol': symbols[i]}) 
                                for i in range(len(symbols))])
        rows, cols = np.where(matrix == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)
        graph = wl(G)
        return graph

def wl(graph):
        #The function is to turn input graph into wl-graph by wl conversion
        node_symbols = nx.get_node_attributes(graph, 'symbol')
        num_iterations = 3
        for _ in range(num_iterations):
            new_symbols = {}
            for node in graph.nodes():
                symbol = node_symbols[node]
                neighbor_symbols = [node_symbols[neighbor] for neighbor in graph.neighbors(node)]
                combined_symbol = symbol + ''.join(sorted(neighbor_symbols))
                new_symbols[node] = combined_symbol
            node_symbols = new_symbols

        new_graph = nx.Graph()
        for node, symbol in node_symbols.items():
            new_graph.add_node(node, symbol=symbol)

        return new_graph
