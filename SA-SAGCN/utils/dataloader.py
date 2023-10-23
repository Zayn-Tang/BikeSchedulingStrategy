import os
import numpy as np
import torch

def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0,1,3), keepdims=True)
    std = train.std(axis=(0,1,3), keepdims=True)
    print('mean.shape:',mean.shape)
    print('std.shape:',std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm



class CikybikeDataset:
    def __init__(self, dataset , params):
        self.x = dataset["arr_0"]
        self.y = dataset["arr_1"]
        self.y = np.expand_dims(self.y, axis=-1)

        self.num_of_history = params.num_of_history
        self.num_of_predict = params.num_of_predict

        self.time = self.x.shape[0]
        # self.init_times = times[slice(input_window_size-1, -output_window_size-1)] 

    def __len__(self):
        return self.time - self.num_of_history - self.num_of_predict

    def __getitem__(self, idx):
        x = self.x[idx, ...]
        y = self.y[idx, ...]
        
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        return x, y


