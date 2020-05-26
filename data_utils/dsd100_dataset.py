import glob
import numpy as np
import torch
import random

class DSD100Dataset(torch.utils.data.Dataset):
    def __init__(self, data_num, sample_len=None, transform=None, folder_type=None):
        self.data_num = data_num
        self.transform = transform
        self.npzs_path = glob.glob('../data/DSD100npz/{0}/*'.format(folder_type))
        self.sample_len = sample_len
        
    def __len__(self):
        return self.data_num
        
    def __getitem__(self, _):
        path = random.sample(self.npzs_path, 1)
        npz_obj = np.load(path[0])
        
        # for debug
        # path = self.npzs_path[7]
        # npz_obj = np.load(path)
        #['bass', 'drums', 'other', 'vocals']
        mixture = npz_obj['mixture']
        bass = npz_obj['bass']
        drums = npz_obj['drums']
        other = npz_obj['other']
        vocals = npz_obj['vocals']
        
        if self.transform:
            pass
        
        if self.sample_len:
            return mixture[:self.sample_len], bass[:self.sample_len], drums[:self.sample_len], other[:self.sample_len], vocals[:self.sample_len]
        else:
            return mixture, bass, drums, other, vocals
        