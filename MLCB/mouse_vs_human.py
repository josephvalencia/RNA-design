import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def compare(mouse_file,human_file):

    human_df = pd.read_csv(human_file,sep='\t')
    mouse_df = pd.read_csv(mouse_file,sep='\t')

    human_df['species'] = 'human'
    mouse_df['species'] = 'mouse'
    joint = pd.concat([human_df,mouse_df])
    joint = joint[joint['Split'] == 'valid']
    print(joint.head())
    sns.histplot(data=joint,x='Stability',hue='species',stat='density',common_norm=False)
    plt.savefig('mrna_stability.png')

if __name__ == "__main__":

    compare('data/saluki_agarwal_kelley/human/genes.tsv','data/saluki_agarwal_kelley/mouse/genes.tsv')