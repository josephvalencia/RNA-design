import torch
from seqopt.oracle_maximizer import OracleMaximizer
from seqopt.oracle import NucleotideDesigner
import matplotlib.pyplot as plt
from scipy.signal import convolve
import numpy as np
import seaborn as sns
import math
import sys
import argparse
import pandas as pd
from functools import partial

def report_example(trajectories,
           name,
           max_train_val,
           property_name,
           cols=3,
           smooth_window=100):

    n = len(trajectories.keys())
    cols = min(cols,n)
    rows = math.ceil(n/cols)
    if len(trajectories.keys()) == 1:
        plt.figure(figsize=(8,2))
        results = list(trajectories.values())[0]
        results = [-x for x in results]
        smoothed_results = convolve(results,np.ones(smooth_window),mode="valid") / smooth_window
        smoothed_domain = [smooth_window+x for x in range(smoothed_results.size)] 
        max = np.max(results)
        begin = results[0]
        delta = max - begin
        plt.plot(smoothed_domain,smoothed_results,c='blue')
        plt.plot(results,alpha=0.2,c='grey')
        plt.axhline(begin,c='black',linestyle='--')
        plt.axhline(max_train_val,c='red',linestyle='--')
        report = f'{begin:.3f}->{max:.3f}={delta:.3f} ({smoothed_results[-1]:.3f})' 
        plt.title('Gibbs with Gradients') 
        sns.despine()

    else: 
        fig,ax = plt.subplots(rows,cols,figsize=(8,3),sharex=True,sharey=True)    
        for ax,(trial,results) in zip(ax.ravel(),trajectories.items()):
            results = [-x for x in results]
            smoothed_results = convolve(results,np.ones(smooth_window),mode="valid") / smooth_window
            smoothed_domain = [smooth_window+x for x in range(smoothed_results.size)] 
            max = np.max(results)
            begin = results[0]
            delta = max - begin
            ax.plot(smoothed_domain,smoothed_results,c='blue')
            ax.plot(results,alpha=0.2,c='grey')
            ax.axhline(begin,c='black',linestyle='--')
            ax.axhline(max_train_val,c='red',linestyle='--')
            report = f'{begin:.3f}->{max:.3f}={delta:.3f} ({smoothed_results[-1]:.3f})' 
            ax.set_title(trial) 
            sns.despine(ax=ax)
        
    #fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel(property_name,fontsize=14)
    plt.xlabel("# Iterations",fontsize=14)
    print(f'saving {name}') 
    plt.tight_layout() 
    plt.savefig(name)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=str)
    parser.add_argument('--property',type=str,default='Mean Ribosome Load')
    parser.add_argument('--max_train_val',type=float,default=1.164601)
    parser.add_argument('--max_test_val',type=float,default=1.164601)
    parser.add_argument('--n_samples',type=int,default=32) 
    parser.add_argument('--n_iter',type=int,default=1000)
    parser.add_argument('--tol',type=int,default=500)
    parser.add_argument('--n_trials',type=int,default=10)
    parser.add_argument('--results_name',type=str,default='trial_results.csv')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',type=int,default=-1)
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--lr',type=float,default=0.001)
    return parser.parse_args()

def oracle_maximizer_template(designer,args):
    ''' Most arguments are determined by args, so fix those'''
    
    device = "cpu" if args.device < 0 else f"cuda:{args.device}"
    template = partial(OracleMaximizer,
                        num_classes=designer.num_classes,
                        class_dim=designer.class_dim,
                        oracles = designer.oracles(),
                        loss_items=[(None,None,1.0)],
                        onehot_fn=designer.onehot_encode,
                        readable_fn=designer.dense_decode,
                        device=device, 
                        optimizer=args.optimizer,
                        learning_rate=args.lr,
                        n_samples=args.n_samples,
                        max_iter=args.n_iter)
    return template

def setup_model_from_lightning(module,device):
    '''Extract PyTorch model from LightningModule and prepare for design'''

    # CuDNN must be turned off for backprop through RNN layers in eval mode 
    torch.backends.cudnn.enabled = False
    model = module.model
    # freeze all the parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device) 
    return model

def lower_confidence_bound(model,*args,**kwargs): 
    # lower confidence bound on the property of interest for EvidentialRegression
    dist = model(*args,**kwargs)
    return dist.mean - dist.epistemic_uncertainty

def run_all_trials(designer : NucleotideDesigner,
                  args : argparse.Namespace):

    template = oracle_maximizer_template(designer,args)

    storage = []
    for i in range(args.n_trials): 
        grad_trajectories = {}
        # sample a random sequence 
        seed_sequence = designer.seed_sequence()
        seed_as_nuc = designer.dense_decode(seed_sequence)
        
        '''
        short_names={'normal': 'STE','softmax':'STE-SM',
                     'gumbel_softmax':'STE-GS','reinforce':'REINFORCE'}
        # all the optimization modes 
        for norm in ['instance', None]: 
            for grad in ['normal','softmax','gumbel_softmax','reinforce']: 
                maximizer = template(seed_sequence,mode='optimize',grad=grad,norm=norm)
                best_seq,improvement,results = maximizer.fit(max_iter=args.n_iter,
                                                             stalled_tol=args.tol)
                best_as_nuc = designer.dense_decode(best_seq)
                name = f'PR-{short_names[grad]}' if norm is None else f'PR-{short_names[grad]}-norm'
                entry = {'seq' : i, 'trial' :  name, 'difference' : -improvement.item(),
                        'initial_loss' : -results[0], 'final_loss' : -results[-1], 
                        'optimized_seq' : best_as_nuc, 'original_seq' : seed_as_nuc}
                storage.append(entry)
                grad_trajectories[name] = results
        report_example(grad_trajectories,
                       f'plots/PR-{i}.pdf',
                       args.max_train_val,
                       args.property,
                       cols=4)
        ''' 
        
        mc_trajectories = {} 
        mcmc_names={'gibbs_with_gradients': 'MCMC-GWG',
                    'path_auxiliary':'MCMC-PA',
                    'langevin':'MCMC-DMALA'}
        # all the sampling modes 
        #for mcmc in ['gibbs_with_gradients','path_auxiliary']:
        #for mcmc in ['gibbs_with_gradients','langevin']:
        for mcmc in ['gibbs_with_gradients']:
            maximizer = template(seed_sequence,mode='sample',mcmc=mcmc)
            best_seq,improvement,results = maximizer.fit(max_iter=args.n_iter,stalled_tol=args.tol)
            best_as_nuc = designer.dense_decode(best_seq) 
            name = mcmc_names[mcmc] 
            entry = {'seq' : i, 'trial' : name, 'difference' : -improvement.item(),
                     'initial_loss' : -results[0], 'final_loss' : -results[-1], 
                      'optimized_seq' : best_as_nuc, 'original_seq' : seed_as_nuc}
            mc_trajectories[name] = results 
            storage.append(entry)
        report_example(mc_trajectories,
                       f'plots/MCMC-{i}.pdf',
                       args.max_train_val,
                       args.property,
                       cols=4)

    summarize_trials(storage,args.results_name,args.property)

def summarize_trials(storage,name,property):

    df = pd.DataFrame(storage)
    df.to_csv(f'{name}.csv')
    sns.boxplot(y='trial',x='difference',data=df)
    sns.despine()
    plt.xlabel(f'Improvement in {property}')
    plt.tight_layout()
    plt.savefig(f'plots/{name}.pdf')
    plt.close()

def tune_langevin(designer : NucleotideDesigner,
                  args : argparse.Namespace):

    template = oracle_maximizer_template(designer,args)
    
    storage = []
    trajectories = {}
    for i in range(args.n_trials): 
        seed_sequence = designer.seed_sequence()
        as_nuc = designer.dense_decode(seed_sequence)
        # all the sampling modes 
        for lr in [1.0,5.0,10,50,100,1000]:
            for beta in [0.5,0.7,0.8,0.9,0.95,0.99]:
                for eps in [1,0,0.1,0.01,0.001,0.0001,0.00001]:
                    maximizer = template(seed_sequence,
                                         mode='sample',
                                         mcmc='langevin',
                                         learning_rate=lr,
                                         beta=beta,
                                         eps=eps)
                    best_seq,improvement,results = maximizer.fit(max_iter=args.n_iter,stalled_tol=args.tol)
                    best_as_nuc = designer.dense_decode(best_seq) 
                    trial = f'langevin-{lr}-{beta}-{eps}' 
                    entry = {'trial' :  trial, 'difference' : improvement.item(),
                              'optimized_seq' : best_as_nuc, 'original_seq' : as_nuc}
                    storage.append(entry)
                    trajectories[trial] = results
            maximizer = template(seed_sequence,
                                 mode='sample',
                                 mcmc='langevin',
                                 learning_rate=lr)
            best_seq,improvement,results = maximizer.fit(max_iter=args.n_iter,stalled_tol=args.tol)
            best_as_nuc = designer.dense_decode(best_seq) 
            trial = f'langevin-{lr}' 
            entry = {'trial' :  trial, 'difference' : improvement.item(),
                      'optimized_seq' : best_as_nuc, 'original_seq' : as_nuc}
            storage.append(entry)
            trajectories[trial] = results
        report_example(trajectories,
                       f'plots/langevin_trial-{i}.pdf',
                       args.max_train_val,
                       args.property,
                       cols=4)

    summarize_trials(storage,args.results_name)
