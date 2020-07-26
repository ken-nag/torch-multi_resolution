import glob
import numpy as np
import torch
import random

class DSD100Dataset(torch.utils.data.Dataset):
    def __init__(self, data_num, sample_len=None, transform=None, folder_type=None, shuffle=True, device=None, augmentation=True):
        self.data_num = data_num
        self.dtype= torch.float32
        self.audio_folder_path = glob.glob('../data/DSD100/{0}/*'.format(folder_type))
        self.sample_len = sample_len
        self.shuffle = shuffle
        self.folder_type = folder_type
        self.device = device
        self.augmentation = augmentation
        
    def __len__(self):
        return self.data_num
    
    def _crop_per_segment(self, x):
        x_len = x.shape[0]
        batch_size = np.ceil(x_len / self.sample_len).astype(np.int32)
        pad_len = self.sample_len - (x_len % self.sample_len)
        pad_x = torch.zeros(x_len + pad_len, dtype=self.dtype, device=self.device)
        pad_x[:x_len] = x[:]
        return pad_x.reshape(batch_size, self.sample_len)
    
    def _random_scaling(self, source):
        scale_coeff = random.randrange(0, 10)/10.0 + 0.25#range 0.25~1.25
        return scale_coeff * source
        
    def _random_chunking(self, source):
         start = np.random.randint(source - self.sample_len)
         return source[start:(start+self.sample_len)]
    
    def _random_mixing(self):
        pass
    
    def _augmentation(self):
        pass
    
    
    def __getitem__(self, idx):
        if self.shuffle:
            path = random.sample(self.npzs_path, 1)
            npz_obj = np.load(path[0])
        else:
            path = self.npzs_path[idx]
            npz_obj = np.load(path)
            
        mixture = torch.from_numpy(npz_obj['mixture']).to(self.dtype).clone()
        bass = torch.from_numpy(npz_obj['bass']).to(self.dtype).clone()
        drums = torch.from_numpy(npz_obj['drums']).to(self.dtype).clone()
        other = torch.from_numpy(npz_obj['other']).to(self.dtype).clone()
        vocals = torch.from_numpy(npz_obj['vocals']).to(self.dtype).clone()
        
        if self.folder_type == 'train' or self.folder_type == 'validation':
            return mixture[:self.sample_len], bass[:self.sample_len], drums[:self.sample_len], other[:self.sample_len], vocals[:self.sample_len]
        
        if self.folder_type == 'test':
            mixture = self._crop_per_segment(mixture)
            bass = self._crop_per_segment(bass)
            drums = self._crop_per_segment(drums)
            other = self._crop_per_segment(other)
            vocals = self._crop_per_segment(vocals)
            
            return mixture, bass, drums, other, vocals
            
            
        
        