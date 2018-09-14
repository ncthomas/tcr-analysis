"""Fundamental functions to summarise TCR repertoire metrics"""

__author__ = 'niclas'

import os

import pandas
import numpy
import scipy.stats
import seaborn
import networkx
import matplotlib
import matplotlib.pyplot
import sklearn.manifold

import Levenshtein

def read_sample(filename):
    """Reads a .cdr3 file into a DataFrame object

    :param filename: Path to .cdr3 file to load
    :type filename: str

    :returns: pandas.DataFrame
    """

    data = pandas.read_csv(filename, header=None, names=['seq','count'])

    return data

def read_all_samples(folder):
    """Reads a set of .cdr3 files into one DataFrame object

    :param folder: Folder to .cdr3 files to load
    :type folder: str

    :returns: pandas.DataFrame
    """

    all_files = os.listdir(folder)
    cdr3_files = [f for f in all_files if '.cdr3' in f]

    data = pandas.DataFrame()

    for file in cdr3_files:

        f = read_sample(folder+file)
        f['sample'] = file.split(".cdr3")[0]

        total = f['count'].sum()
        f['percent'] = 100*f['count']/total
        data = data.append(f)

    data = data.sort_values('count', ascending=False)

    return data


def get_sample_names(data):
    """Obtains all sample names in a dataset

    :param data: DataFrame containing sequences, counts and sample names
    :type data: pandas.DataFrame

    :returns: list
    """

    names = numpy.sort(data['sample'].unique())

    return names

def plot_frequency_distribution(data, samples, output_folder):
    """Plots freqeuency distribution of sequences in each sample

    :param data: DataFrame containing sequences, counts and sample names
    :type data: pandas.DataFrame
    :param samples: List of strings of sample names
    :type samples: list
    :param output_folder: Directory in which to save figures
    :type output_folder: str

    :returns: None
    """

    for samp in samples:

        f = data[data['sample']==samp].sort_values('count', ascending=False)

        counts = pandas.DataFrame(f['count'].value_counts()).reset_index()
        total = counts['count'].sum()
        counts['percent'] = 100*counts['count']/total
        
        fig, ax = matplotlib.pyplot.subplots()
        ax.scatter(x=counts['index'], y=counts['percent'], color='black', s=20)
        ax.set_xlabel('Number of Copies of CDR3')
        ax.set_ylabel('% Total Repertoire')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim()
        seaborn.despine()
        matplotlib.pyplot.savefig(output_folder+samp+'_freq_distribution.png', bbox_inches='tight')
        matplotlib.pyplot.close()

def get_jaccard(x, y):
    """Returns Jaccard index for two samples x and y

    :param x: List or pandas.Series containing CDR3 sequences
    :type x: list or pandas.Series
    :param y: List or pandas.Series containing CDR3 sequences
    :type y: list or pandas.Series

    :returns: float
    """

    union = len(set(x).union(set(y)))
    intersection = len(set(x).intersection(set(y)))

    return intersection/union

def calculate_pairwise_jaccard_indexes(data, samples):
    """Calculates Jaccard indexes of all pairwise samples

    :param data: DataFrame containing sequences, counts and sample names
    :type data: pandas.DataFrame
    :param samples: List of strings of sample names
    :type samples: list

    :returns: pandas.DataFrame
    """

    jaccard_results = pandas.DataFrame({'sample1':[],'sample2':[],'jaccard':[]})

    for i in range(len(samples)):
        for j in range(len(samples)):
            s1 = data[data['sample']==samples[i]]['seq']
            s2 = data[data['sample']==samples[j]]['seq']

            df = pandas.DataFrame({'sample1':[samples[i]],
                               'sample2':[samples[j]],
                               'jaccard':[get_jaccard(s1,s2)]})

            jaccard_results = jaccard_results.append(df)

    jaccard_results = jaccard_results.reset_index().drop('index', axis=1)

    return jaccard_results

def plot_samples_jaccard_index(data, indexes, output_path):
    """Plots frequency distribution of sequences in each sample

    :param data: DataFrame containing sequences, counts and sample names
    :type data: pandas.DataFrame
    :param indexes: DataFrame containing sequences, counts and sample names
    :type indexes: list
    :param output_path: DataFrame containing sequences, counts and sample names
    :type output_path: str

    :returns: None
    """

    filtered_data = data[data.index.isin(indexes)].sort_values('jaccard', ascending=False)

    fig, ax = matplotlib.pyplot.subplots(figsize=(2.5,5))
    seaborn.barplot(x='label', y='jaccard', data=filtered_data, color='black')
    ax.set_xlabel('')
    ax.set_ylabel('Jaccard Index')
    seaborn.despine()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    matplotlib.pyplot.savefig(output_path+'_jaccard_index.png', bbox_inches='tight')
    matplotlib.pyplot.close()

def pairwise_jaccard_heatmap(data, samples, output_folder):
    """Calculates Jaccard indexes of all pairwise samples

    :param data: DataFrame containing sequences, counts and sample names
    :type data: pandas.DataFrame
    :param samples: List of strings of sample names
    :type samples: list

    :returns: pandas.DataFrame
    """

    jaccard_results = numpy.zeros([len(samples), len(samples)])

    for i in range(len(samples)):
        for j in range(len(samples)):
            
            s1 = data[data['sample']==samples[i]]['seq']
            s2 = data[data['sample']==samples[j]]['seq']

            jaccard_results[i,j] = get_jaccard(s1,s2)
            
    df_jaccard = pandas.DataFrame(jaccard_results, columns=samples, index=samples)
    
    fig, ax = matplotlib.pyplot.subplots()
    ax = seaborn.heatmap(df_jaccard, cmap='bwr')
    matplotlib.pyplot.savefig(output_folder+'_jaccard_heatmap.png', bbox_inches='tight')
    matplotlib.pyplot.close()
    
def get_shannon_entropy(data, samples):
    """Calculates Shannon entropy for each sample

    :param data: DataFrame containing sequences, counts and sample names
    :type data: pandas.DataFrame
    :param samples: List of strings of sample names
    :type samples: list

    :returns: pandas.DataFrame
    """

    shannon = pandas.DataFrame({'sample':[], 'shannon':[]})

    for s in samples:
        shannon = shannon.append({'sample':s, 'shannon':scipy.stats.entropy(data[data['sample']==s]['count'])}, ignore_index=True)

    return shannon

def match_CDR3s_to_known_specificities(seqs, specs):
    """Joins CDR3 sequencing data to known specificities

    :param seqs: TCR CDR3 sequencing data output from Decombinator
    :type seqs: pandas.DataFrame
    :param specs: Known CDR3 specificities
    :type specs: pandas.DataFrame

    :returns: pandas.DataFrame
    """

    def apply_color(x):

        if x is numpy.nan:
            col = 'black'
        else:
            col = 'red'

        return col

    seqs_specs = pandas.merge(seqs, specs, how='left', left_on='seq', right_on='CDR3')
    seqs_specs['color'] = seqs_specs['Epitope species'].apply(lambda x: apply_color(x))

    return seqs_specs

def plot_frequency_distribution_with_specificities(seq_specs, output_path, top_n=50):
    """Joins CDR3 sequencing data to known specificities

    :param seq_specs: TCR CDR3 sequencing data output from Decombinator
    :type seq_specs: pandas.DataFrame
    :param output_path: Known CDR3 specificities
    :type output_path: str
    :param top_n: Only the top_n most frequenct CDR3s will be shown in the figure
    :type top_n: int

    :returns: pandas.DataFrame
    """

    samples = seq_specs['sample'].unique()

    for s in samples:

        df = seq_specs[seq_specs['sample']==s].sort_values('count', ascending=False).head(top_n)
        df = df.fillna('')
        df = df.reset_index().drop('index', axis=1)

        ## DISTRIBUTION PLOT
        ####################
        fig, ax = matplotlib.pyplot.subplots(figsize=(10,10))
        ax = seaborn.barplot(x='seq', y='percent', data=df, palette=df['color'])
        for index, row in df.iterrows():
            ax.text(row.name,
                    -0.01,
                    row['Epitope species'],
                    color='red',
                    ha="center",
                    va='top',
                    rotation=90)
        ax.set_xlabel('')
        ax.set_ylabel('CDR3 Frequency (%)')
        ax.set_xticklabels([])
        seaborn.despine()
        matplotlib.pyplot.savefig(output_path+s+'_dist_with_specificity.png', bbox_inches='tight')
        matplotlib.pyplot.close()

        
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
    
    networkx.draw_networkx_nodes(G, pos, node_size=counts, alpha=1, node_color='black')
    networkx.draw_networkx_edges(G, pos, alpha=0.4, width=3, edge_color='red')
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.savefig(output_path+'_network.png', bbox_inches='tight')
