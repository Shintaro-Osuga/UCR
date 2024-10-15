import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RegDataloader(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return torch.tensor(self.df.iloc[idx].values)