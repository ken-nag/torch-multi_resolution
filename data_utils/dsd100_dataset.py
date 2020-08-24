import glob
import numpy as np
import torch
import random
import torchaudio

class DSD100Dataset(torch.utils.data.Dataset):
    def __init__(self, data_num, full_data_num=None, sample_len=None, transform=None, folder_type=None, shuffle=True, device=None, augmentation=None):
        self.data_num = data_num
        self.dtype= torch.float32
        self.full_data_num = full_data_num
        self.folder_type = folder_type
        
        if self.folder_type == 'Test':
            self.audio_folder_path = glob.glob('../data/DSD_Dsampled/Sources/Test/*')
            
        if self.folder_type=='Dev':
            self.bass_folder_path = glob.glob('../data/DSD_Dev/bass/*')
            self.drums_folder_path = glob.glob('../data/DSD_Dev/drums/*')
            self.vocals_folder_path = glob.glob('../data/DSD_Dev/vocals/*')
            self.other_folder_path = glob.glob('../data/DSD_Dev/other/*')
        
        if self.folder_type=='Validation':
            self.bass_folder_path = glob.glob('../data/DSD_Dev/bass/*')[-self.full_data_num:]
            self.drums_folder_path = glob.glob('../data/DSD_Dev/drums/*')[-self.full_data_num:]
            self.vocals_folder_path = glob.glob('../data/DSD_Dev/vocals/*')[-self.full_data_num:]
            self.other_folder_path = glob.glob('../data/DSD_Dev/other/*')[-self.full_data_num:]
            
        self.sample_len = sample_len
        self.shuffle = shuffle
        self.device = device
        self.augmentation = augmentation
        
    def __len__(self):
        return self.data_num
      
    def _random_scaling(self, source):
        scale_coeff = random.uniform(0, 1.25) #range 0.25~1.25
        return scale_coeff * source
        
    # def _random_chunking(self, source):
    #      start = np.random.randint(len(source) - self.sample_len)
    #      return source[start:(start+self.sample_len)]
     
    # def _augmentation(self,source):
    #     chunked = self._random_chunking(source)
    #     return self._random_scaling(chunked)
            
    def __getitem__(self, idx):
        if (self.folder_type == 'Dev') or (self.folder_type == 'Validation'):
            if self.augmentation:
                bass_path = random.sample(self.bass_folder_path, 1)[-1]
                drums_path = random.sample(self.drums_folder_path, 1)[-1]
                other_path = random.sample(self.other_folder_path, 1)[-1]
                vocals_path = random.sample(self.vocals_folder_path, 1)[-1]
            else:
                path = self.audio_folder_path[idx]    
                bass_path = self.bass_folder_path[idx]
                drums_path = self.drums_folder_path[idx]
                other_path = self.other_folder_path[idx]
                vocals_path = self.vocals_folder_path[idx]
                    
        elif self.folder_type == 'Test':
            path = self.audio_folder_path[idx]      
            bass_path = path + '/bass.wav'
            drums_path = path + '/drums.wav'
            other_path = path + '/other.wav'
            vocals_path = path + '/vocals.wav'
                
        bass, _ = torchaudio.load(bass_path)
        drums, _ = torchaudio.load(drums_path)
        other, _ = torchaudio.load(other_path)
        vocals, _ = torchaudio.load(vocals_path)
        
        bass = bass.squeeze(0).to(self.dtype).to(self.device)
        drums = drums.squeeze(0).to(self.dtype).to(self.device)
        other = other.squeeze(0).to(self.dtype).to(self.device)
        vocals = vocals.squeeze(0).to(self.dtype).to(self.device)
        
        if (self.folder_type == 'Dev') or (self.folder_type == 'Validation'):
            if self.augmentation:
                bass = self._random_scaling(bass)
                drums = self._random_scaling(drums)
                other = self._random_scaling(other)
                vocals = self._random_scaling(vocals)
                
            start = np.random.randint(len(bass) - self.sample_len)
            bass = bass[start:start+self.sample_len]
            drums = drums[start:start+self.sample_len]
            other = other[start:start+self.sample_len]
            vocals = vocals[start:start+self.sample_len]
                
        mixture = bass + drums + other + vocals
            
        return mixture, bass, drums, other, vocals
            
            
        
        