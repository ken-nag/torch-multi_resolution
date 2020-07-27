import glob
import numpy as np
import torch
import random
import torchaudio

class DSD100Dataset(torch.utils.data.Dataset):
    def __init__(self, data_num, sample_len=None, transform=None, folder_type=None, shuffle=True, device=None, augmentation=None):
        self.data_num = data_num
        self.dtype= torch.float32
        self.audio_folder_path = glob.glob('../data/DSD_Dsampled/Sources/{0}/*'.format(folder_type))
        self.sample_len = sample_len
        self.shuffle = shuffle
        self.folder_type = folder_type
        self.device = device
        self.augmentation = augmentation
        
    def __len__(self):
        return self.data_num
    
    # def _crop_per_segment(self, x):
    #     x_len = x.shape[0]
    #     batch_size = np.ceil(x_len / self.sample_len).astype(np.int32)
    #     pad_len = self.sample_len - (x_len % self.sample_len)
    #     pad_x = torch.zeros(x_len + pad_len, dtype=self.dtype, device=self.device)
    #     pad_x[:x_len] = x[:]
    #     return pad_x.reshape(batch_size, self.sample_len)
    
    def _random_scaling(self, source):
        scale_coeff = random.uniform(0, 1.25) #range 0.25~1.25
        return scale_coeff * source
        
    def _random_chunking(self, source):
         start = np.random.randint(len(source) - self.sample_len)
         return source[start:(start+self.sample_len)]
     
    def _augmentation(self,source):
        chunked = self._random_chunking(source)
        return self._random_scaling(chunked)
            
    def __getitem__(self, idx):
        if self.folder_type == 'Dev':
            #random mixiking
            paths = random.sample(self.audio_folder_path, 4)
            bass_path, _ = paths[0] + '/bass.wav'
            drums_path = paths[1] + '/drums.wav'
            other_path = paths[2] + '/other.wav'
            vocals_path = paths[3] + '/vocals.wav'
        elif self.folder_type == 'Test':
            path = self.audio_folder_path[idx]      
            bass_path, _ = path + '/bass.wav'
            drums_path, _ = path + '/drums.wav'
            other_path, _ = path + '/other.wav'
            vocals_path, _ = path + '/vocals.wav'
                
        bass, _ = torchaudio.load(bass_path)
        drums, _ = torchaudio.load(drums_path)
        other, _ = torchaudio.load(other_path)
        vocals, _ = torchaudio.load(vocals_path)
        
        bass = bass.squeeze(0).to(self.dtype)
        drums = drums.squeeze(0).to(self.dtype)
        other = other.squeeze(0).to(self.dtype)
        vocals = vocals.squeeze(0).to(self.dtype)
        
        if self.folder_type == 'Dev':
            if self.augmentation:
                bass = self._augmentation(bass)
                drums = self._augmentation(drums)
                other = self._augmentation(other)
                vocals = self._augmentation(vocals)
            else:
                start = np.random.randint(len(bass) - self.sample_len)
                bass = bass[start:start+self.sample_len]
                drums = drums[start:start+self.sample_len]
                other = other[start:start+self.sample_len]
                vocals = vocals[start:start+self.sample_len]
                
        mixture = bass + drums + other + vocals
            
        return mixture, bass, drums, other, vocals
            
            
        
        