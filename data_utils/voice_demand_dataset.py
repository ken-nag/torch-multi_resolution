# -*- coding: utf-8 -*-
import glob
import torch
import torchaudio
import os
import random
import numpy as np

class VoicebankDemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_num, full_data_num=None, folder_type=None, sample_len=None, shuffle=True, device=None, augmentation=None):
        self.dtype = torch.float32
        self.dataset_root = '../data/VoicebankDemand/'
        self.sample_len = sample_len
        self.folder_type = folder_type
        self.data_num = data_num
        self.shuffle = shuffle
        self.device = device
        self.augmentation = augmentation
        
        if self.folder_type == 'train' or self.folder_type == 'validation':
            self.clean_root = self.dataset_root + '/clean_trainset_wav/'
            self.noisy_root = self.dataset_root + '/noisy_trainset_wav/'
              
        if self.folder_type == 'test':
            self.clean_root = self.dataset_root + '/clean_testset_wav/'
            self.noisy_root = self.dataset_root + '/noisy_testset_wav/'
        
        file_path = glob.glob(self.clean_root + '*.wav')
        wav_names = [os.path.split(e)[-1] for e in file_path]

        if self.folder_type == 'train':
            self.wav_names = wav_names[:full_data_num]
        
        elif self.folder_type == 'validation':
            self.wav_names = wav_names[-full_data_num:]

        else:
            self.wav_names = wav_names
            print(self.wav_names)
            
        
    def _cut_or_pad(self, x):
        x_len = x.shape[-1]
        
        if x_len >= self.sample_len:
            x = x[:self.sample_len]
        else:
            x = self._zero_pad(x)
            
        return x
    
    def _crop_or_pad(self, x):
        x_len = x.shape[-1]
        
        if x_len >= self.sample_len:
            x = self._crop_per_segment(x)
        else:
            print("detect x < {0}".format(self.sample_len))
            x = self._zero_pad(x)
            x = x.unsqueeze(0)
        return x
    
    def _crop_per_segment(self, x):
        x_len = x.shape[0]
        batch_size = torch.ceil(torch.tensor(x_len / self.sample_len)).to(torch.int32)
        pad_len = self.sample_len - (x_len % self.sample_len)
        pad_x = torch.zeros(x_len + pad_len, dtype=self.dtype, device=self.device)
        pad_x[:x_len] = x[:]
        return pad_x.reshape(batch_size, self.sample_len)
    
    def _zero_pad(self, x):
        x_len = x.shape[-1]
        pad_x = torch.zeros(self.sample_len, dtype=self.dtype)
        pad_x[:x_len] = x[:]
        return pad_x
    
    def _random_chunk_or_pad(self, noisy, clean):
        x_len = len(noisy)
        if x_len > self.sample_len:
            start = np.random.randint(x_len - self.sample_len)
            clean = clean[start:start+self.sample_len]
            noisy = noisy[start:start+self.sample_len]
        else:
            clean = self._zero_pad(clean)
            noisy = self._zero_pad(noisy)
        return noisy, clean
    
    def _random_snr(self, noisy, clean):
        noise = noisy - clean
        p_noise = noise.pow(2).mean().sqrt()
        p_clean = clean.pow(2).mean().sqrt()
        snr = random.uniform(-5,15)
        k = p_clean/(torch.tensor(10, dtype=torch.float32).to(self.device).pow(snr/20.0)*p_noise)
        noise = k*noise
        return clean + noise
    
    def _swap_noise(self, clean):
        wav_name = random.sample(self.wav_names, 1)[-1]
        clean2, _ = torchaudio.load(self.clean_root+wav_name)
        noisy2, _ = torchaudio.load(self.noisy_root+wav_name)
        clean2 = clean2.squeeze(0).to(self.dtype)
        noisy2 = noisy2.squeeze(0).to(self.dtype)
        noise = noisy2 - clean2
        
        noise_len = len(noise)
        clean_len = len(clean)
        if clean_len >= noise_len:
            clean = clean[:noise_len]
            noisy = clean + noise
        else:
            noisy = clean + noise[:clean_len]
        return  noisy, clean
    
    def __len__(self):
        return self.data_num
        
    def __getitem__(self, idx):
        if self.shuffle:
            wav_name = random.sample(self.wav_names, 1)[-1]
        else:
            print(idx)
            wav_name = self.wav_names[idx]
            print(wav_name)
            
        clean, _ = torchaudio.load(self.clean_root+wav_name)
        noisy, _ = torchaudio.load(self.noisy_root+wav_name)

        clean = clean.squeeze(0).to(self.dtype)
        noisy = noisy.squeeze(0).to(self.dtype)
        
        if self.folder_type == 'train' or self.folder_type == 'validation':
            if self.augmentation:
                noisy, clean = self._swap_noise(clean)
                noisy = self._random_snr(noisy, clean)
            else:
                clean = self._cut_or_pad(clean)
                noisy = self._cut_or_pad(noisy)
                 
        return noisy, clean    


