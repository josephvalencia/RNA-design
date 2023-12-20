import pandas as pd
import numpy as np

def base_to_idx(base):
    if base == 'A':
        return 0
    elif base == 'C':
        return 1
    elif base == 'G':
        return 2
    elif base == 'T':
        return 3
    else:
        raise ValueError(f'base {base} not recognized')

def analyze_toeholds(fname):
    df = pd.read_csv(fname)
    switches = df['switch_sequence'].tolist()
    triggers = df['trigger_sequence'].tolist()
    
    counts = np.zeros((59,4))
    for switch in switches:
        assert len(switch) == 59
        for i,base in enumerate(switch):
            counts[i,base_to_idx(base)] += 1
    print(counts / len(switches))



if __name__ == '__main__':
    analyze_toeholds('data/toehold_valeri_etal/downsampled_pruned_old_data.txt')


