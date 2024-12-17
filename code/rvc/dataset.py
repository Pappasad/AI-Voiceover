import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class VoiceDataset(Dataset):

    def __init__(self, source_dir, target_dir):
        self.sources = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.npy')]
        self.targets = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.npy')]

    def __len__(self):
        return min(len(self.sources), len(self.targets))
    
    def __getitem__(self, index):
        source_mel_spec = np.load(self.sources[index])
        target_mel_spec = np.load(self.targets[index])
        return torch.tensor(source_mel_spec, dtype=torch.float32), torch.tensor(target_mel_spec, dtype=torch.float32)