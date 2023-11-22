import torch
from models.optimus_5prime.model import MeanRibosomeLoadModule
from optseq.oracle_maximizer import OracleMaximizer
import matplotlib.pyplot as plt
from scipy.signal import convolve
import numpy as np
from models.optimus_5prime.data import *
import seaborn as sns
import math
import sys

def encode_seq(seq):
    utr = torch.nn.functional.one_hot(seq,num_classes=4).permute(0,2,1).float().requires_grad_(True)
    return utr

def to_nucleotide(dense):                                                                                                                                        
    as_list = dense.cpu().numpy().ravel().tolist()
    mapping = 'ACGT'
    nucs = [mapping[x] for x in as_list] 
    return ''.join(nucs)

def report(trajectories,name,cols=3):

    n = len(trajectories.keys())
    cols = min(cols,n)
    rows = math.ceil(n/cols)
    fig,ax = plt.subplots(rows,cols,figsize=(8,3),sharex=True,sharey=True)    
    
    for ax,(trial,results) in zip(ax.ravel(),trajectories.items()):
        
        results = [-x for x in results]
        smooth_window = 100
        smoothed_results = convolve(results,np.ones(smooth_window),mode="valid") / smooth_window
        smoothed_domain = [smooth_window+x for x in range(smoothed_results.size)] 
        max = np.max(results)
        begin = results[0]
        delta = max - begin
        ax.plot(smoothed_domain,smoothed_results,c='blue')
        ax.plot(results,alpha=0.2,c='grey')
        ax.axhline(begin,c='red',linestyle='--')
        ax.axhline(1.164601,c='black',linestyle='--')
        report = f'{begin:.3f}->{max:.3f}={delta:.3f} ({smoothed_results[-1]:.3f})' 
        #ax.text(0.05, 0.9,report,transform=ax.transAxes)
        ax.set_title(trial) 
        sns.despine(ax=ax)
    
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Mean Ribosome Load",fontsize=14)
    plt.xlabel("# Iterations",fontsize=14)
    print(f'saving {name}') 
    plt.tight_layout() 
    plt.savefig(name)
    plt.close()

def plot_results(results,name,smooth_window=100):

    smoothed_results = convolve(results,np.ones(smooth_window),mode="valid") / smooth_window
    smoothed_domain = [smooth_window+x for x in range(smoothed_results.size)] 
    plt.plot(results,alpha=0.2,c='grey')
    plt.plot(smoothed_domain,smoothed_results,c='blue')
    plt.tight_layout()
    plt.savefig(f'plots/{name}.png')
    plt.close()

def grad_trial(model):

    n_trials = 100
    n_iter = 2000

    for i in range(n_trials): 
        seed_sequence = torch.randint(0,4,(1,50))
        as_nuc = to_nucleotide(seed_sequence)
        # all the optimization modes 
        maximizer = OracleMaximizer(seed_sequence,num_classes=4,
                                    class_dim=1,
                                    oracles = [model],
                                    loss_items=[(None,None,1.0)],
                                    onehot_fn=encode_seq,
                                    readable_fn=to_nucleotide,
                                    device=device, 
                                    mode='optimize',optimizer='adam',grad='normal')
        best_seq,improvement,results = maximizer.fit(max_iter=n_iter,stalled_tol=tol)

        plot_results(results,f'grad-{i}')

def run_trials(model,device):

    n_iter = 10000
    tol = 2000
    n_trials = 10
    test_idx = -1 
    storage = []
    for i in range(n_trials): 
        grad_trajectories = {}
        seed_sequence = torch.randint(0,4,(1,50))
        as_nuc = to_nucleotide(seed_sequence)
        
        # lower confidence bound on the property of interest
        def fwd_method(*args,**kwargs): 
            dist = model(*args,**kwargs)
            return dist.mean - dist.epistemic_uncertainty
        short_names={'normal': 'STE','softmax':'STE-SM',
                     'gumbel_softmax':'STE-GS','reinforce':'REINFORCE'}
        # all the optimization modes 
        for norm in ['instance', None]: 
            for grad in ['normal','softmax','gumbel_softmax','reinforce']: 
                maximizer = OracleMaximizer(seed_sequence,num_classes=4,
                                            class_dim=1,
                                            oracles = [fwd_method],
                                            loss_items=[(None,None,1.0)],
                                            onehot_fn=encode_seq,
                                            readable_fn=to_nucleotide,
                                            device=device, 
                                            mode='optimize',
                                            optimizer='adam',
                                            grad=grad,
                                            norm=norm,
                                            n_samples=32)
                best_seq,improvement,results = maximizer.fit(max_iter=n_iter,stalled_tol=tol)
                name = f'PR-{short_names[grad]}' if norm is None else f'PR-{short_names[grad]}-norm'
                entry = {'trial' :  name, 'difference' : -improvement.item(),
                        'initial_loss' : -results[0], 'final_loss' : -results[-1], 
                        'optimized_seq' : to_nucleotide(best_seq), 'original_seq' : as_nuc}
                storage.append(entry)
                grad_trajectories[name] = results
        report(grad_trajectories,f'plots/PR-{i}.pdf',cols=4)
        
        '''

        mc_trajectories = {} 
        mcmc_names={'gibbs_with_gradients': 'MCMC-GWG',
                    'path_auxiliary':'MCMC-PA',
                    'langevin':'MCMC-DMALA'}
        # all the sampling modes 
        #for mcmc in ['gibbs_with_gradients','path_auxiliary']:
        for mcmc in ['path_auxiliary','gibbs_with_gradients']:
            maximizer = OracleMaximizer(seed_sequence,num_classes=4,
                                        class_dim=1,
                                       oracles = [fwd_method],
                                        loss_items=[(None,None,1.0)],
                                        onehot_fn=encode_seq,
                                        readable_fn=to_nucleotide,
                                        device=device, 
                                        mode='sample',mcmc=mcmc,
                                        max_iter=n_iter)
            best_seq,improvement,results = maximizer.fit(max_iter=n_iter,stalled_tol=n_iter)
            name = mcmc_names[mcmc] 
            entry = {'trial' : name, 'difference' : -improvement.item(),
                     'initial_loss' : -results[0], 'final_loss' : -results[-1], 
                      'optimized_seq' : to_nucleotide(best_seq), 'original_seq' : as_nuc}
            mc_trajectories[name] = results 
            storage.append(entry)
        report(mc_trajectories,f'plots/MCMC-{i}.pdf',cols=4)
        '''

    df = pd.DataFrame(storage)
    df.to_csv('trial_results.csv')
    sns.boxplot(y='trial',x='difference',data=df)
    plt.tight_layout()
    plt.savefig('violinplot.pdfs')
    plt.close()

    sns.lmplot(data=df,x='initial_loss',y='final_loss',hue='trial',
                col='trial',col_wrap=4)
    plt.tight_layout()
    plt.savefig('lmplot.png')
    plt.close()


def run_trial_langevin(model,device):

    n_iter = 10000 
    tol = 2000
    n_trials = 10
    test_idx = -1 
    storage = []
    trajectories = {}
    # lower confidence bound on the property of interest
    def fwd_method(*args,**kwargs): 
        dist = model(*args,**kwargs)
        return dist.mean - dist.epistemic_uncertainty
    for i in range(n_trials): 
        seed_sequence = torch.randint(0,4,(1,50))
        as_nuc = to_nucleotide(seed_sequence)
        # all the sampling modes 
        #for lr in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
        for lr in [0.001,0.1,0.5,1.0,5.0,10,100,1000]:
            maximizer = OracleMaximizer(seed_sequence,num_classes=4,
                                        class_dim=1,
                                        oracles = [fwd_method],
                                        loss_items=[(None,None,1.0)],
                                        onehot_fn=encode_seq,
                                        readable_fn=to_nucleotide,
                                        device=device, 
                                        mode='sample',mcmc='langevin',
                                        max_iter=n_iter,learning_rate=lr)
            best_seq,improvement,results = maximizer.fit(max_iter=n_iter,stalled_tol=tol)
            trial = f'langevin-{lr}' 
            entry = {'trial' :  trial, 'difference' : improvement.item(),
                      'optimized_seq' : to_nucleotide(best_seq), 'original_seq' : as_nuc}
            storage.append(entry)
            trajectories[trial] = results

        report(trajectories,f'plots/langevin_trial-{i}.pdf',cols=4)

    df = pd.DataFrame(storage)
    df.to_csv('langevin_results.csv')
    sns.boxplot(y='trial',x='difference',data=df)
    plt.tight_layout()
    plt.savefig('boxplot_langevin.pdf')
    plt.close()

if __name__ == "__main__":

    gpu = int(sys.argv[1])
    device = "cpu" if gpu < 0 else f"cuda:{gpu}"
    chkpt_dir = "/home/bb/valejose/valejose/TE-design/models/optimus_5prime/checkpoints/"
    module = MeanRibosomeLoadModule.load_from_checkpoint(
                #f"{chkpt_dir}radiant-bush-15-epoch=10-val_loss=0.1286.ckpt",map_location=device)
                f"{chkpt_dir}olive-field-17-epoch=11-val_loss=0.2742.ckpt",map_location=device) 
    
    model = module.model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device) 
    print('MRL model loaded')
    
    run_trials(model,device)
    #run_trial_langevin(model,device)
