import networkx
import  matplotlib
import matplotlib.pyplot

import tcranalysis.io
import tcranalysis.graphs

CHAIN = 'beta'
DATA_DIRECTORY = '/Users/laurapallett/Documents/lauren/data/' + CHAIN + '/'
OUTPUT_DIRECTORY = '/Users/laurapallett/Documents/lauren/output/' + CHAIN +'/'
MAX_SEQS = 50
MAX_WEIGHT = 1
N_GRAMS = 1
mw = 10


def get_sample_status(x):
    """Convenience function to create column with sample status"""

    status_int = int(x.split("_")[2])

    if status_int <= 3:
        status = 'CTRL'
    elif 3 < status_int <= 6:
        status = 'DIABETES'
    else:
        status = 'NULL'

    return status

def apply_color(x):
    """Convenience function to create a column of strings of colors"""

    if x is np.nan:
        col = 'black'
    else:
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
# value w - analysis is run for values between 0 and 9
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
    
    if int(k.split('_')[2]) <= 3:
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
ax.legend(custom_lines, ['CTRL', 'DIABETES'])
matplotlib.pyplot.show()
