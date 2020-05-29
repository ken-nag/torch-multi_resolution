# -*- coding: utf-8 -*-
import glob
import torch
import torchaudio
import os
import random


class VoicebankDemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_num, folder_type=None, sample_len=None, shuffle=True):
        self.dtype = torch.float32
        self.dataset_root = '../data/VoicebankDemand/'
        self.sample_len = sample_len
        self.folder_type = folder_type
        self.data_num = data_num
        self.shuffle = shuffle
        
        if self.folder_type == 'train' or self.folder_type == 'validation':
            self.clean_root = self.dataset_root + '/clean_trainset_wav/'
            self.noisy_root = self.dataset_root + '/noisy_trainset_wav/'
              
        if self.folder_type == 'test':
            self.clean_root = self.dataset_root + '/clean_testset_wav/'
            self.noisy_root = self.dataset_root + '/noisy_testset_wav/'
        
        file_path = glob.glob(self.clean_root + '*.wav')
        self.wav_names = [os.path.split(e)[-1] for e in file_path]
    
    
    def _cut_or_pad(self, x):
        x_len = x.shape[-1]
        
        if x_len >= self.sample_len:
            x = x[:self.sample_len]
        else:
            x = self._zero_pad(x)
            
        return x

    def _zero_pad(self, x):
        x_len = x.shape[-1]
        pad_x = torch.zeros(self.sample_len, dtype=self.dtype)
        pad_x[:x_len] = x[:]
        return pad_x
        
    def __len__(self):
        return self.data_num
        
    def __getitem__(self, idx):
        if self.shuffle:
            wav_name = random.sample(self.wav_names, 1)[-1]
        else:
            wav_name = self.wav_name[idx]
            
        clean, _ = torchaudio.load(self.clean_root+wav_name)
        noisy, _ = torchaudio.load(self.noisy_root+wav_name)
        
        clean = clean.squeeze(0).to(self.dtype)
        noisy = noisy.squeeze(0).to(self.dtype)
        
        clean = self._cut_or_pad(clean)
        noisy = self._cut_or_pad(noisy)
    
        return noisy, clean    


