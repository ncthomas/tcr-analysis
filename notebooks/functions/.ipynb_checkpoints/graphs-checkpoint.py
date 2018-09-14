"""Network analysis module based on NetworkX"""

__author__ = 'niclas'

import numpy
import networkx
import matplotlib
import matplotlib.pyplot
import sklearn.manifold

import Levenshtein
        
def get_distances(seqs):
    """Calculates pairwise distances between CDR3 sequences

    :param seqs: TCR CDR3 sequencing data output from Decombinator
    :type seqs: list

    :returns: numpy.array
    """
    
    num_seqs = len(seqs)
    dists = numpy.zeros([num_seqs,num_seqs])
    
    for i in range(num_seqs):
        for j in range(num_seqs):
            dists[i,j] = Levenshtein.distance(seqs[i],seqs[j])
        
    return dists

def get_dimension_mapping(dists):
    """Joins CDR3 sequencing data to known specificities

    :param dists: Array of pairwise distances, [i,j]th entry represents Levenshtein distance between sequence i and j
    :type dists: numpy.array

    :returns: numpy.array
    """
    
    mds = sklearn.manifold.MDS(n_components=2, random_state=123, dissimilarity='precomputed')
    coords = mds.fit_transform(dists)
    
    return coords

def get_graph(seqs, dists, max_weight):
    """Joins CDR3 sequencing data to known specificities

    :param seqs: TCR CDR3 sequencing data output from Decombinator
    :type seqs: list
    :param dists: Array of pairwise distances, [i,j]th entry represents Levenshtein distance between sequence i and j
    :type dists: numpy.array
    :param max_weight: Distances up to max_weight will be joined in the graph G
    :type max_weight: int

    :returns: pandas.DataFrame
    """
        
    G = networkx.Graph()
    num_seqs = len(seqs)

    for i in range(num_seqs):
        for j in range(num_seqs):
            if dists[i,j]<=max_weight:
                G.add_edge(seqs[i], seqs[j], weight=dists[i,j])
            
    edges = [(u, v) for (u, v, d) in G.edges(data=True)]

    return G

def plot_graph(G, coords, seqs, counts, output_path):
    """Plots a graph of CDR3 sequencing data where points represent CDR3 sequences, point sizes represent their
    corresponding counts, and edges between points are created if two sequences are similar.

    :param G: networkx graph representing CDR3 sequence similarities
    :type G: networkx.graph
    :param coords: 2D representation of CDR3 sequences, projected into 2D using multidimensional scaling
    :type coords: numpy.array
    :param seqs: TCR CDR3 sequencing data output from Decombinator
    :type seqs: list
    :param counts: TCR CDR3 sequence counts
    :type counts: list
    :param output_path: Directory in which to save figure
    :type output_path: str

    :returns:
    """
        
    pos = {seqs[i]: coords[i] for i in range(len(seqs))}
    
    fig, ax = matplotlib.pyplot.subplots(figsize=(10,10))
    networkx.draw_networkx_nodes(G, pos, node_size=counts, alpha=1, node_color='black')
    networkx.draw_networkx_edges(G, pos, alpha=0.4, width=3, edge_color='red')
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.savefig(output_path+'_network.png', bbox_inches='tight')

