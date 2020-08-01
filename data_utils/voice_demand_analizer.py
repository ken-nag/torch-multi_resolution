import torch
import torchaudio
import glob
import os
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

class Analizer():
    def __init__(self, folder_type):
        self.dtype = torch.float32
        self.dataset_root = '../data/VoicebankDemand/'
        self.folder_type = folder_type
        
        if self.folder_type == 'train':
            self.clean_root = self.dataset_root + '/clean_trainset_wav/'
            self.noisy_root = self.dataset_root + '/noisy_trainset_wav/'
              
        if self.folder_type == 'test':
            self.clean_root = self.dataset_root + '/clean_testset_wav/'
            self.noisy_root = self.dataset_root + '/noisy_testset_wav/'
        
        file_path = glob.glob(self.clean_root + '*.wav')
        self.wav_names = [os.path.split(e)[-1] for e in file_path]

    def _cal_snr(self, noisy, clean):
        noise = noisy - clean
        p_clean = clean.pow(2).mean().sqrt()
        p_noise = noise.pow(2).mean().sqrt()
        snr = 20*torch.log10(p_clean/p_noise)
        return snr
    
    def run(self,):
        print(self.folder_type)
        snr_list = []
        for wav_name in self.wav_names:
            clean, _ = torchaudio.load(self.clean_root+wav_name)
            noisy, _ = torchaudio.load(self.noisy_root+wav_name)
            clean = clean.squeeze(0).to(self.dtype)
            noisy = noisy.squeeze(0).to(self.dtype)
            snr = self._cal_snr(noisy, clean)
            snr_list.append(snr)
            
            
        
        snr_list = torch.tensor(snr_list, dtype=torch.float32)
        print('mean:', snr_list.mean())
        print('max:', snr_list.max())
        print('mix:', snr_list.min())
        print('median:', snr_list.median())
        
        snr_list = snr_list.detach().clone().cpu().numpy()
        plt.hist(snr_list, bins=100)
        
        
        
        
if __name__ == '__main__':
    train_obj = Analizer(folder_type='train')
    test_obj = Analizer(folder_type='test')
    
    train_obj.run()
    test_obj.run()