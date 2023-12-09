import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_diff():

    storage = []
    for fname in [f'group{i}.csv' for i in range(1,5)]:
        print(fname) 
        df = pd.read_csv(fname)
        storage.append(df)
    combined = pd.concat(storage)
    combined['seq_id'] = [f'seq_{i}_{t}' for i,t in zip(combined['seq'],combined['trial'])]
    combined['norm'] = combined['trial'].str.endswith('norm')
    combined['trial'] = [t.replace('-norm','') for t in combined['trial']]
    print(combined['trial'].unique())
    combined.to_csv('long_runs.csv') 
    combined = combined[~combined['trial'].str.startswith('MCMC')]
    sns.boxplot(data=combined,x='difference',y='trial',hue='norm')
    plt.ylabel('Sampling algorithm')
    plt.xlabel('Improvement in mean ribosome load')
    sns.despine()
    plt.tight_layout()
    plt.savefig('long_difference_boxplot.pdf')
    plt.close()


if __name__ == "__main__":

    plot_diff()

