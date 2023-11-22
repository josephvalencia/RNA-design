import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns

def plot_results(csv):

    df = pd.read_csv(csv)
    
    g = sns.boxplot(data=df,x='difference',y='trial')
    sns.despine() 
    plt.tight_layout()
    plt.savefig('boxplot.png')
    plt.close()

if __name__ == "__main__":

    plot_results(sys.argv[1])
