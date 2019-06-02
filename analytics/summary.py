import pandas
import matplotlib.pyplot
import seaborn
import numpy

import tcranalysis.io
import tcranalysis.metrics
import tcranalysis.graphs

CHAIN = 'beta'
DATA_DIRECTORY = '/Users/laurapallett/Documents/leo/data/' + CHAIN + '/'
OUTPUT_DIRECTORY = '/Users/laurapallett/Documents/leo/output/002/' + CHAIN +'/'
MAX_SEQS = 500
MAX_WEIGHT = 1
N_GRAMS = 1

def get_sample_status(x):
    """Convenience function to create column with sample status"""

    status = x.split("_")[3]

    return status


def apply_color(x):
    """Convenience function to create a column of strings of colors"""

    if x is np.nan:
        col = 'black'
    else:
        col = 'red'

    return col

# Get data

data = tcranalysis.io.read_all_samples(DATA_DIRECTORY)

samples = tcranalysis.io.get_sample_names(data)
tcranalysis.metrics.plot_frequency_distribution(data, samples, OUTPUT_DIRECTORY)
tcranalysis.metrics.pairwise_jaccard_heatmap(data, samples, OUTPUT_DIRECTORY)


# Shannon Entropy

shannon_results = tcranalysis.metrics.get_shannon_entropy(data, samples)
shannon_results['status'] = shannon_results['sample'].apply(lambda x: get_sample_status(x))

fig, ax = matplotlib.pyplot.subplots(figsize=(2.5, 5))
seaborn.swarmplot(x='status', y='shannon', data=shannon_results, size=10, hue='status')
ax.set_xlabel('')
ax.set_ylabel('Shannon Entropy')
ax.legend().set_visible(False)
seaborn.despine()
matplotlib.pyplot.savefig(OUTPUT_DIRECTORY+'shannon.png', bbox_inches='tight')
matplotlib.pyplot.close()

# Compare CDR3s in samples with known specificities from VDJdb

specs = pandas.read_csv('../resources/specificities.tsv', sep='\t')[['Gene','CDR3','V','J','MHC A','Epitope species']]
specs = specs.replace('HomoSapiens', numpy.nan)

if CHAIN == 'alpha':
    specs = specs[(specs['Gene'] == 'TRA') & (specs['MHC A'] == 'HLA-A*02')]
elif CHAIN == 'beta':
    specs = specs[(specs['Gene'] == 'TRB') & (specs['MHC A'] == 'HLA-A*02')]

seq_specs = tcranalysis.metrics.match_CDR3s_to_known_specificities(data, specs)
tcranalysis.metrics.plot_frequency_distribution_with_specificities(seq_specs, OUTPUT_DIRECTORY, top_n=50)


