import networkx
import matplotlib
import matplotlib.pyplot

import tcranalysis.io
import tcranalysis.graphs

CHAIN = 'beta'
DATA_DIRECTORY = '/Users/laurapallett/Documents/leo/data/' + CHAIN + '/'
OUTPUT_DIRECTORY = '/Users/laurapallett/Google Drive/laura/tcr_sequencing/wp005/' + CHAIN +'/'
MAX_SEQS = 50
MAX_WEIGHT = 1
N_GRAMS = 1
mw = 10
topn = 5
sample1 = 'dcr_EXO_0070_NEG_' + CHAIN
sample2 = 'dcr_EXO_0070_POS_' + CHAIN


def get_sample_status(x):
    """Convenience function to create column with sample status"""

    status = x.split("_")[3]

    return status


def apply_color(x, samples):
    """Convenience function to create a column of strings of colors"""

    if x == samples[0]:
        col = 'black'
    elif x == samples[1]:
        col = 'red'

    return col


data = tcranalysis.io.read_all_samples(DATA_DIRECTORY)

samples = tcranalysis.io.get_sample_names(data)

for i in range(len(samples)):

    sample_data = data[data['sample']==samples[i]]
    seqs = sample_data['seq'].tolist()[0:MAX_SEQS]
    counts = sample_data['count'].tolist()

    dists = tcranalysis.graphs.get_distances(seqs)
    coords = tcranalysis.graphs.get_dimension_mapping(dists)

    G = tcranalysis.graphs.get_graph(seqs, dists, max_weight=MAX_WEIGHT)
    tcranalysis.graphs.plot_graph(G, coords, seqs, counts, OUTPUT_DIRECTORY+samples[i])


# Calculate the "connectedness" of each sample
# i.e. how many distinct components (subgraphs) exist
# when edges are drawn between a pair of CDR3 sequences
# only if the Levenshtein distance is less than a given
#  value w - analysis is run for values between 0 and 9
# and these values are shown on the axis in the plot below

connectedness = {}

for i in range(len(samples)):

    sample_data = data[data['sample']==samples[i]]
    seqs = sample_data['seq'].tolist()
    counts = sample_data['count'].tolist()

    dists = tcranalysis.graphs.get_distances(seqs)

    diversity = {}

    for w in range(0,mw):

        G = tcranalysis.graphs.get_graph(seqs, dists, max_weight=w)
        num_components = len(list(networkx.connected_components(G)))

        diversity[w] = num_components/len(seqs)

    x = [k for k in diversity.keys()]
    y = [v for v in diversity.values()]

    connectedness[samples[i]] = (x,y)


fig, ax = matplotlib.pyplot.subplots(figsize=(10,10))

for k, v in connectedness.items():

    if k.split('_')[3] == 'POS':
        x = v[0]
        y = v[1]
        ax.plot(x, y, color='black', lw=2)
    else:
        x = v[0]
        y = v[1]
        ax.plot(x, y, color='red', lw=2)

ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set(xlabel='Levenshtein Distance Joining CDR3 Sequences', ylabel='Proportion of Components in Graph')
custom_lines = [matplotlib.lines.Line2D([0], [0], color='black', lw=2),
                matplotlib.lines.Line2D([0], [0], color='red', lw=2)]
ax.legend(custom_lines, ['POS', 'NEG'])
matplotlib.pyplot.show()

# Plot paired graph

samples = [sample1, sample2]

sample_data = data[data['sample'].isin(samples)]
sample_data['color'] = sample_data['sample'].apply(lambda x: apply_color(x, samples))

seqs = sample_data['seq'].tolist()
colors = sample_data['color'].tolist()
counts = sample_data['count'].tolist()

dists = tcranalysis.graphs.get_distances(seqs)
coords = tcranalysis.graphs.get_dimension_mapping(dists)

G = tcranalysis.graphs.get_graph(seqs, dists, max_weight=MAX_WEIGHT)

tcranalysis.graphs.plot_paired_graph(G, coords, seqs, counts, colors, topn, OUTPUT_DIRECTORY)