import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
from src.datasets.augmentation import AddGaussianNoise
from audiomentations import SpecFrequencyMask


class LibriSamples(Dataset):

    def __init__(self, data_path, partition= "train", freq_mask=False, add_noise=False): # You can use partition to specify train or dev

        assert partition in ['train', 'dev']

        self.partition = partition
        self.freq_mask = SpecFrequencyMask(p=0.5) if freq_mask else None
        self.add_noise = AddGaussianNoise(p=0.5) if add_noise else None

        self.X_dir = os.path.join(data_path, f'{partition}/mfcc') 
        self.Y_dir = os.path.join(data_path, f'{partition}/transcript') 

        file_names = os.listdir(self.X_dir)

        self.X_files = [os.path.join(self.X_dir, f) for f in file_names] 
        self.Y_files = [os.path.join(self.Y_dir, f) for f in file_names] 

        self.LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', \
                            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', \
                            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']

        assert(len(self.X_files) == len(self.Y_files))

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, idx):
    
        X = np.load(self.X_files[idx]) 
        Y = np.load(self.Y_files[idx]) 

        # Cepstral Mean Normalization
        X = (X - X.mean(axis=0))/(X.std(axis=0)+1e-7)
        # torchaudio.transforms.SlidingWindowCmn(cmn_window: int = 600, min_cmn_window: int = 100, center: bool = False, norm_vars: bool = False)

        if self.freq_mask:
            X = self.freq_mask(X)

        if self.add_noise:
            X = self.add_noise(X)
        
        X = torch.from_numpy(X)
        Yy = torch.LongTensor([self.LETTER_LIST.index(yy) for yy in Y]) 

        return X, Yy
    
    def collate_fn(batch):

        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]

        batch_x_pad = pad_sequence(batch_x, batch_first=True) 
        lengths_x = [len(x) for x in batch_x] 

        batch_y_pad = pad_sequence(batch_y, batch_first=True) 
        lengths_y = [len(y) for y in batch_y] 

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)


class LibriSamplesTest(Dataset):

    def __init__(self, data_path, test_order):
        test_dir = os.path.join(data_path, 'test')
        test_order_df = pd.read_csv(os.path.join(test_dir, test_order))
        test_order_list = test_order_df.file.to_list() 
        test_data_path = os.path.join(test_dir, 'mfcc')
        self.X_files = [os.path.join(test_data_path, x) for x in test_order_list] 
    
    def __len__(self):
        return len(self.X_files)
    
    def __getitem__(self, idx):
        X = torch.from_numpy(np.load(self.X_files[idx]))
        # Cepstral Mean Normalization
        X = (X - X.mean(axis=0))/(X.std(axis=0)+1e-7)
        return X
    
    def collate_fn(batch):
        batch_x = [x for x in batch]
        batch_x_pad = pad_sequence(batch_x, batch_first=True) 
        lengths_x = [len(x) for x in batch_x] 

        return batch_x_pad, torch.tensor(lengths_x)