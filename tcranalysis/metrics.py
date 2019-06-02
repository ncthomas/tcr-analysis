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
        
        fig, ax = matplotlib.pyplot.subplots(figsize=(10,10))
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


def pairwise_jaccard_heatmap(data, samples, output_folder):

    """Calculates Jaccard indexes of all pairwise samples

    :param data: DataFrame containing sequences, counts and sample names
    :type data: pandas.DataFrame
    :param samples: List of strings of sample names
    :type samples: list
    :param output_folder: target directory to save image
    :type output_folder: string

    :returns: pandas.DataFrame
    """

    jaccard_results = numpy.zeros([len(samples), len(samples)])

    for i in range(len(samples)):
        for j in range(len(samples)):
            
            s1 = data[data['sample']==samples[i]]['seq']
            s2 = data[data['sample']==samples[j]]['seq']

            jaccard_results[i,j] = get_jaccard(s1,s2)
            
    df_jaccard = pandas.DataFrame(jaccard_results, columns=samples, index=samples)    
    masked_df_jaccard = df_jaccard.mask(numpy.triu(numpy.ones(df_jaccard.shape)).astype(bool))

    fig, ax = matplotlib.pyplot.subplots(figsize=(10,10))
    ax = seaborn.heatmap(masked_df_jaccard, cmap='bwr')
    matplotlib.pyplot.savefig(output_folder+'jaccard_heatmap.png', bbox_inches='tight')
    matplotlib.pyplot.close()
    
    return df_jaccard
    
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