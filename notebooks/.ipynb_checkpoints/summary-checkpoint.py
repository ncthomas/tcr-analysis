import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.insert(0, '../')
from tcranalysis import functions
import config

sns.set(style='white', font_scale=1)


def get_sample_person(x):
    """Convenience function to create column with sample name"""

    person = x.split("_")[2]

    return person


def get_sample_status(x):
    """Convenience function to create column with sample status (POS or NEG)"""

    status = x.split("_")[3]

    return status


def get_sample_chain(x):
    """Convenience function to create column with sample chain (alpha or beta)"""

    chain = x.split("_")[4].split(".")[0]

    return chain


def apply_color(x):
    """Convenience function to create a column of strings of colors"""

    if x is np.nan:
        col = 'black'
    else:
        col = 'red'

    return col

data = functions.read_all_samples(config.Config.DATA_DIRECTORY)

## Distribution of frequencies
##############################

samples = functions.get_sample_names(data)
functions.plot_frequency_distribution(data, samples, config.Config.OUTPUT_DIRECTORY)

## Jaccard Index
################

jaccard_results = functions.calculate_pairwise_jaccard_indexes(data, samples)
jaccard_results['sample1_name'] = jaccard_results['sample1'].apply(lambda x: get_sample_person(x))
jaccard_results['sample1_status'] = jaccard_results['sample1'].apply(lambda x: get_sample_status(x))
jaccard_results['sample1_chain'] = jaccard_results['sample1'].apply(lambda x: get_sample_chain(x))
jaccard_results['sample2_name'] = jaccard_results['sample2'].apply(lambda x: get_sample_person(x))
jaccard_results['sample2_status'] = jaccard_results['sample2'].apply(lambda x: get_sample_status(x))
jaccard_results['sample2_chain'] = jaccard_results['sample2'].apply(lambda x: get_sample_chain(x))
jaccard_results['label'] = jaccard_results['sample1'] + jaccard_results['sample2']

functions.plot_samples_jaccard_index(jaccard_results, [41, 24], config.Config.OUTPUT_DIRECTORY)
functions.plot_samples_jaccard_index(jaccard_results, [55, 20], config.Config.OUTPUT_DIRECTORY)

## Shannon Entropy
##################

shannon_results = functions.get_shannon_entropy(data, samples)

shannon_results['name'] = shannon_results['sample'].apply(lambda x: get_sample_person(x))
shannon_results['status'] = shannon_results['sample'].apply(lambda x: get_sample_status(x))
shannon_results['chain'] = shannon_results['sample'].apply(lambda x: get_sample_chain(x))

# alpha

fig, ax = plt.subplots(figsize=(2.5, 5))
sns.swarmplot(x='status', y='shannon', data=shannon_results[shannon_results['chain'] == 'alpha'], size=10)
sns.despine()
ax.set_xlabel('')
ax.set_ylabel('Shannon Entropy')
plt.savefig(config.Config.OUTPUT_DIRECTORY + 'shannon.png', bbox_inches='tight')
plt.close()

# beta

fig, ax = plt.subplots(figsize=(2.5, 5))
sns.swarmplot(x='status', y='shannon', data=shannon_results[shannon_results['chain'] == 'beta'], size=10)
sns.despine()
ax.set_xlabel('')
ax.set_ylabel('Shannon Entropy')
plt.savefig(config.Config.OUTPUT_DIRECTORY + 'shannon.png', bbox_inches='tight')
plt.close()

## Compare CDR3s in samples with known specificities from VDJdb

specs = pd.read_csv('../data/specificities.tsv', sep='\t')[['Gene', 'CDR3', 'V', 'J', 'MHC A', 'Epitope species']]
specs = specs.replace('HomoSapiens', np.nan)
alpha_specs = specs[(specs['Gene'] == 'TRA') & (specs['MHC A'] == 'HLA-A*02')]
beta_specs = specs[(specs['Gene'] == 'TRB') & (specs['MHC A'] == 'HLA-A*02')]

data['name'] = data['sample'].apply(lambda x: get_sample_person(x))
data['status'] = data['sample'].apply(lambda x: get_sample_status(x))
data['chain'] = data['sample'].apply(lambda x: get_sample_chain(x))

alpha_data = data[data['chain'] == 'alpha']
beta_data = data[data['chain'] == 'beta']

alpha_seq_specs = functions.match_CDR3s_to_known_specificities(alpha_data, alpha_specs)
beta_seq_specs = functions.match_CDR3s_to_known_specificities(beta_data, beta_specs)

functions.plot_frequency_distribution_with_specificities(alpha_seq_specs, config.Config.OUTPUT_DIRECTORY, top_n=50)

## Which specificities are in each sample?

alpha_samples = alpha_seq_specs['sample'].unique()

for s in alpha_samples:
    df = pd.DataFrame(alpha_seq_specs[alpha_seq_specs['sample'] == s]['Epitope species'].value_counts()).reset_index()
    df.columns = ['Epitope', 'Count']

    fig, ax = plt.subplots(figsize=(2.5, 5))
    ax = sns.barplot(x='Epitope', y='Count', data=df, color='black')
    ax.set_xlabel('')
    ax.set_ylabel('Epitope-Specific CDR3 Count')
    ax.set_yticks([i for i in range(df['Count'].max() + 1)])
    sns.despine()
    plt.savefig(config.Config.OUTPUT_DIRECTORY + s + '_epitope_cdr3_counts.png')

beta_samples = beta_seq_specs['sample'].unique()

for s in beta_samples:
    df = pd.DataFrame(beta_seq_specs[beta_seq_specs['sample'] == s]['Epitope species'].value_counts()).reset_index()
    df.columns = ['Epitope', 'Count']

    fig, ax = plt.subplots(figsize=(2.5, 5))
    ax = sns.barplot(x='Epitope', y='Count', data=df, color='black')
    ax.set_xlabel('')
    ax.set_ylabel('Epitope-Specific CDR3 Count')
    ax.set_yticks([i for i in range(df['Count'].max() + 1)])
    sns.despine()
    plt.savefig(config.Config.OUTPUT_DIRECTORY + s + '_epitope_cdr3_counts.png')