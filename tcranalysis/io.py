"""Input/Output module to read files produced by Decombinator"""

__author__ = 'niclas'

import os

import pandas
import numpy

def read_sample(filename):
    """Reads a .cdr3 file into a DataFrame object

    :param filename: Path to .cdr3 file to load
    :type filename: str

    :returns: pandas.DataFrame
    """

    data = pandas.read_csv(filename, header=None, names=['seq', 'count'])

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
