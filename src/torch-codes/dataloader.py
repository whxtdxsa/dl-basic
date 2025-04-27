import torch
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    def __init__(self, x, t):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx]
