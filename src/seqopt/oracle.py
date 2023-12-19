import torch
from abc import ABC, abstractmethod
from typing import Callable, List, Union

class NucleotideDesigner(ABC):

    def __init__(self,num_classes,class_dim):
        self.num_classes = num_classes
        self.class_dim = class_dim

    @abstractmethod
    def onehot_encode(self,seq : torch.Tensor) -> torch.Tensor:
        '''Convert a sequence to a one-hot encoding'''
        pass

    @abstractmethod
    def dense_decode(self,seq) -> Union[str,List[str]] :
        ''' Convert a dense sequence to a readable nucleotide sequence'''
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
