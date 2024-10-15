import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict
from functools import partial
import math

class SimRegModel(nn.Module):
    def __init__(self, in_size:int, out_size:int):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(in_size, 128),
                            nn.Dropout1d(0.1),
                            nn.Linear(128, 1028),
                            nn.Dropout1d(0.1),
                            nn.Linear(1028, 512),
                            nn.Dropout1d(0.1),
                            nn.Linear(512, 64),
                            nn.Dropout1d(0.1),
                            nn.Linear(64, 256),
                            nn.Linear(256, out_size))
        
    def forward(self, x):
        out = self.seq(x)
        return out