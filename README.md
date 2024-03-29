# RNA-design

This repo implements `seqopt`, a lightweight library enabling users to perform model-based optimization using differentiable nucleotide property models written in PyTorch. Its only dependency is `torch>=1.9.0`, so it can be easily installed into your environment via

```
pip install git+https://github.com/josephvalencia/RNA-design.git
```

`seqopt` addresses the problem of using a trained sequence -> property predictor to identify sequences which the model predicts to satisfy a target value. It supports two basic paradigms for sampling: probabilistic reparameterization and gradient-guided Markov Chain Monte Carlo. Both strategies leverage the gradient of the oracle model with respect to its inputs, while avoiding the pathology of evaluating the model on infeasible (i.e. non one-hot) input. 

#### Probabilistic Reparameterization
In this strategy, an auxiliary categorical distribution is defined from which to sample discrete sequences. The distribution parameters are updated in a gradient-ascent-like fashion to optimize the model output function based on the input gradients or function evaluations of sequences sampled from the distribution. `seqopt` supports a variety of algorithms for updating the parameters.
1. Straight-Through Estimator (STE) [(Bengio et al, 2013)](https://arxiv.org/abs/1308.3432)
2. Softmax STE [(Chung et al, 2016)](https://arxiv.org/abs/1609.01704)
3. Gumbel-softmax STE [(Jang et al, 2016)](https://arxiv.org/abs/1611.01144)
4. REINFORCE/score function estimator 

See [(Linder and Seelig, 2021)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04437-5) for a TensorFlow implementation of many of these algorithms and experiments on several nucleotide property tasks.

#### Gradient-guided Markov Chain Monte Carlo
In this strategy, no additional parameters are learned but sequence gradients are used to bias a proposal distribution for MCMC over discrete distributions. Currently, two such approaches are supported.
1. Gibbs with Gradients [(Grathwohl et. al 2020)](https://arxiv.org/abs/2102.04509)
2. Discrete Langevin Proposal [(Zhang et. al 2022)](https://arxiv.org/abs/2206.09914)

## Basic Usage

To equip a Pytorch sequence model with the ability to design sequences, a user must extend the `NucleotideDesigner` [base class](src/seqopt/oracle.py).

```
import torch
from abc import ABC, abstractmethod
from typing import Callable, List, Union

class NucleotideDesigner(ABC):

    def __init__(self,num_classes,class_dim):
        self.num_classes = num_classes
        self.class_dim = class_dim

    @abstractmethod
    def onehot_encode(self,seq : torch.Tensor) -> torch.Tensor:
        '''Convert a dense sequence tensor to a one-hot encoding'''
        pass

    @abstractmethod
    def dense_decode(self,seq : torch.Tensor) -> Union[str,List[str]] :
        ''' Convert a dense sequence tensor to a readable nucleotide sequence'''
        pass

    @property
    @abstractmethod
    def oracles(self) -> List[Union[Callable,torch.nn.Module]]:
        '''Return a list of differentiable oracles that will be applied to the sequence'''
        pass
    
    @abstractmethod
    def seed_sequence(self) -> torch.Tensor:
        '''Generate a random sequence of a given length'''
        pass
```

## Examples
`seqopt` began as part of a project titled "Extrapolative benchmarking of model-based discrete sampling methods for RNA design", presented at the 2023 Machine Learning in Computational Biology conference. Based on prior works, I implemented basic CNN+LSTM models for predicting ribosome load of 5' UTRs, degradation properties of mRNAs, and toehold switch activities. I evaluated the ability of model-based optimization to produce designs which exceed the property values observed during training, as scored by a more powerful model. Code for training these models and performing sequence optimization using the `seqopt` API is located in the [MLCB](MLCB/) folder. See our [extended abstract](assets/MLCB_Discrete_Search_Nov.pdf) and [poster](assets/MLCB_Poster.pdf) for further details.

## Future Development
Naive input optimization can drift towards regions of input space where model predictions are inaccurate. For the experiments above, I implemented a simple uncertainty estimation procedure using [Evidential Regression (Amini et al. 2019)](https://arxiv.org/abs/1910.02600) to permit sampling of sequences with high property values and low uncertainty.

A probably more robust approach is to use a prior density model $p(x)$ of the input nucleotide space to mitigate drift by sampling according to Bayes law, i.e. $\log p(x|y) \propto \log p(y|x) + \log p(x)$, where $p(y|x)$ is a probabilistic property predictor. While `seqopt` permits the use of multiple sequence oracles to implement such a procedure, this is subject to the availability of suitable likelihood models, and testing for this is ongoing.  

