
import numpy as np
import torch
import torch.utils.data as data

class stock_data(data.Dataset):

    def __init__(self, data_array, seq_len, stage):
        self.seq_len = seq_len
        self.stage = stage
        self.data = self.load_stock_data(data_array)

    def __len__(self):
        return len(self.data) - self.seq_len
    
    def load_stock_data(self, data_array):
        train_size = int(np.round(0.7*data_array.shape[0]))
        val_size = int(np.round(0.1*data_array.shape[0]))
        test_size = data_array.shape[0] - (train_size + val_size)
        if self.stage == "train":
            return torch.tensor(data_array[:train_size,:])
        elif self.stage == "val":
            return torch.tensor(data_array[train_size:train_size+val_size,:])
        elif self.stage == "test":
            return torch.tensor(data_array[train_size+val_size:,:])

    def __getitem__(self, idx):
        X = self.data[idx:idx+self.seq_len].float()
        y = self.data[idx+self.seq_len].float()
        return idx, X, y