"""author: Maximilian Glumann"""

import torch
from dataclasses import dataclass, field

@dataclass
class DataParameters:
    pin_memory:bool = True if torch.cuda.is_available() else False
    batch_size:int = 8
    num_workers:int = 20

@dataclass 
class TrainingParameters:
    device:str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs:int = 50

@dataclass 
class OptimizerParameters:
    learning_rate:float = 1e-4

@dataclass
class HyperParameters:
    data:DataParameters = field(default_factory=DataParameters)
    training:TrainingParameters = field(default_factory=TrainingParameters)
    optimizer:OptimizerParameters = field(default_factory=OptimizerParameters)
    
hyperparameters = HyperParameters()